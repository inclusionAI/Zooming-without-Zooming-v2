"""
Processor functions for API-based single sample inference.
"""

import os
import json
import re
try:
    from qwen_vl_utils import fetch_image  # type: ignore
except ImportError:
    # Lightweight local/API mode does not need the full Qwen utility package.
    # Keep the same small input contract used below (``{'image': path_or_url}``).
    from io import BytesIO
    import requests
    from PIL import Image

    def fetch_image(spec):
        source = spec["image"]
        if str(source).startswith(("http://", "https://")):
            response = requests.get(source, timeout=45)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        return Image.open(source).convert("RGB")
from utils.function_call_parser import parse_function_call_response
from utils.context_utils import process_image
from utils.context_utils import pil_to_base64_data_uri
from utils.context_utils import estimate_tokens
from utils.general_qa_tool import compute_score
from utils.result_utils import save_trajectory
from search.tree import SearchNode
from engine.api.api_model_caller import create_model_caller
from engine.api.api_tool_handler import set_global_prompts as set_tool_handler_prompts


# Global prompt variables (will be set by infer_api.py)
SYSTEM_PROMPT = ""
TOOL_CALL_CROP_MULTI_TRUN_PROMPT = ""
FEEDBACK_PROMPT_TOO_SMALL = ""


def set_global_prompts(system_prompt, tool_call_prompt, feedback_too_small):
    """
    Set global prompt variables for API processors.
    Also sets them in tool handler.
    
    Args:
        system_prompt: System prompt string
        tool_call_prompt: Tool call prompt template
        feedback_too_small: Feedback for too small bbox
    """
    global SYSTEM_PROMPT, TOOL_CALL_CROP_MULTI_TRUN_PROMPT, FEEDBACK_PROMPT_TOO_SMALL
    SYSTEM_PROMPT = system_prompt
    TOOL_CALL_CROP_MULTI_TRUN_PROMPT = tool_call_prompt
    FEEDBACK_PROMPT_TOO_SMALL = feedback_too_small
    
    # Also set in tool handler
    set_tool_handler_prompts(feedback_too_small, tool_call_prompt)


# ============================================================================
# Helper Functions - Extract common logic
# ============================================================================

def parse_and_load_multiple_images(sample, args, save_dir):
    """
    Parse <image> placeholders from question text and load all images.
    Supports both single and multiple images with backward compatibility.
    
    Args:
        sample: Data sample with 'problem'/'question' and 'images' fields
        args: Command line arguments with image_folder
        save_dir: Directory to save images
        
    Returns:
        tuple: (image_map, image_list, question_text, user_content_list)
            - image_map: dict mapping image names to PIL.Image objects
            - image_list: list of (image_name, PIL.Image) tuples in order
            - question_text: question text with <image> placeholders removed
            - user_content_list: list of content items for API message (images + text)
        Returns (None, None, None, None) if image loading fails
    """
    question = sample.get('problem', sample.get('question', ''))
    image_paths = sample.get('images', [])
    
    # Count <image> placeholders in question
    image_placeholder_pattern = r'<image>'
    placeholder_count = len(re.findall(image_placeholder_pattern, question, re.IGNORECASE))
    
    # Determine how many images we need
    # Use placeholder count if available, otherwise use number of images in sample
    # If neither exists, default to 1 (backward compatibility)
    if placeholder_count > 0:
        num_images_needed = placeholder_count
    elif len(image_paths) > 0:
        num_images_needed = len(image_paths)
    else:
        num_images_needed = 1  # Default to at least one image for backward compatibility
    
    # Ensure we don't try to load more images than available
    num_images_to_load = min(num_images_needed, len(image_paths)) if len(image_paths) > 0 else 0
    
    if num_images_to_load == 0 and num_images_needed > 0:
        print(f"Warning: Need {num_images_needed} image(s) but none provided in sample")
        return None, None, None, None
    
    # Load images
    image_map = {}
    image_list = []
    user_content_list = []
    
    for i in range(num_images_to_load):
        image_path = os.path.join(args.image_folder, image_paths[i])
        # Naming convention: first image is 'original_image', others are 'original_image_1', 'original_image_2', etc.
        image_name = f'original_image_{i}' if i > 0 else 'original_image'
        
        try:
            original_image = fetch_image({'image': image_path, 'max_pixels': args.max_pixels})
            processed_image = process_image(original_image, args.max_pixels, args.min_pixels)
            # Store original PIL object for potential cropping
            image_map[image_name] = original_image
            image_list.append((image_name, original_image))
            
            # Add to user content for API message
            user_content_list.append({
                "type": "image_url",
                "image_url": {"url": pil_to_base64_data_uri(original_image)}
            })
            
            # Save image with unique filename for multi-image cases
            # For single image: original_image.jpg
            # For multiple images: original_image.jpg, original_image_1.jpg, original_image_2.jpg, etc.
            os.makedirs(save_dir, exist_ok=True)
            if num_images_to_load == 1:
                # Single image: use standard name
                image_file_path = os.path.join(save_dir, "original_image.jpg")
                processed_image.save(image_file_path)
            else:
                # Multiple images: save with unique filename
                if i == 0:
                    # First image: use standard name
                    image_file_path = os.path.join(save_dir, "original_image.jpg")
                else:
                    # Additional images: save with numbered filename
                    image_file_path = os.path.join(save_dir, f"original_image_{i}.jpg")
                processed_image.save(image_file_path)
                if i > 0:
                    # print(f"[Multi-Image] Saved {image_name} to {image_file_path}")
                    pass
            
        except FileNotFoundError:
            print(f"Error: Image not found at {image_path}")
            return None, None, None, None
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None, None, None, None
    
    # Remove <image> placeholders from question text
    question_text = re.sub(image_placeholder_pattern, '', question, flags=re.IGNORECASE).strip()
    # Clean up any extra whitespace/newlines
    question_text = re.sub(r'\n+', '\n', question_text).strip()
    
    # Add text content at the end
    if question_text:
        user_content_list.append({"type": "text", "text": question_text})
    
    return image_map, image_list, question_text, user_content_list


def _initialize_sample_and_image(sample, args, save_dir):
    """
    Initialize sample data and load/process images (supports multiple images).
    
    Returns:
        tuple: (question_id, question, ground_truth, image_map, image_list, question_text, user_content_list)
        Returns None if image loading fails
    """
    question_id = sample.get("doc_id", sample.get("question_id", "N/A"))
    question = sample.get('problem', sample.get('question', ''))
    ground_truth = sample.get("solution")
    
    # Save initial trajectory info
    initial_traj_info = {
        "doc_id": question_id,
        "initial_prompt": question,
        "ground_truth": ground_truth
    }
    save_trajectory(save_dir,initial_traj_info)
    
    # Parse and load multiple images
    image_map, image_list, question_text, user_content_list = parse_and_load_multiple_images(sample, args, save_dir)
    if image_map is None:
        return None
    
    # Save injected system prompt for observability
    try:
        with open(os.path.join(save_dir, '0_injected_system_prompt.txt'), 'w', encoding='utf-8') as f:
            f.write(SYSTEM_PROMPT)
    except Exception:
        pass
    
    return question_id, question, ground_truth, image_map, image_list, question_text, user_content_list


def _create_initial_search_node(question_text, image_map, image_list, user_content_list, system_prompt):
    """
    Create initial SearchNode with conversation history and image map.
    Supports both single and multiple images.
    
    Args:
        question_text: Question text (with <image> placeholders removed)
        image_map: dict mapping image names to PIL.Image objects
        image_list: list of (image_name, PIL.Image) tuples in order
        user_content_list: list of content items for API message (images + text)
        system_prompt: System prompt string
        
    Returns:
        SearchNode: Initialized search node
    """
    # Build text conversation history with image references
    # Format: "[Image: name1]\n[Image: name2]\n{question_text}"
    image_refs = [f"[Image: {name}]" for name, _ in image_list]
    user_content_text = "\n".join(image_refs) + (f"\n{question_text}" if question_text else "")
    
    # Build API conversation history
    # Use user_content_list if provided (contains images + text), otherwise fallback to text only
    if user_content_list:
        api_user_content = user_content_list
    else:
        api_user_content = [{"type": "text", "text": question_text}] if question_text else []
    
    return SearchNode(
        conversation_history=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content_text}
        ],
        api_conversation_history=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": api_user_content
            }
        ],
        image_map=image_map.copy(),
        current_turn=0,
        current_token_count=estimate_tokens(system_prompt + question_text)
    )


def _add_user_message_to_node(node, message_text):
    """
    Add a user message to both conversation histories and update token count.
    
    Args:
        node: SearchNode to update
        message_text: Message text to add
    """
    next_user_message = {
        "role": "user",
        "content": [{"type": "text", "text": message_text}]
    }
    node.api_conversation_history.append(next_user_message)
    node.conversation_history.append({"role": "user", "content": message_text})
    node.current_token_count += estimate_tokens(message_text)


def _extract_responses_from_node(node):
    """
    Extract conversation history and assistant responses from node.
    
    Returns:
        tuple: (conversation_history, assistant_responses)
    """
    conversation_history = node.conversation_history
    assistant_responses = [
        msg['content'] for msg in conversation_history
        if msg['role'] == 'assistant'
    ]
    return conversation_history, assistant_responses


def _evaluate_trajectory(question, ground_truth, conversation_history, assistant_responses, question_id):
    """
    Evaluate trajectory and compute scores.
    
    Returns:
        tuple: (accuracy_score, trajectory_score, trajectory_analysis, trajectory_text)
    """
    accuracy_score = 0.0
    trajectory_score = 0.0
    trajectory_analysis = ""
    trajectory_text = ""
    
    # Generate trajectory text
    for message in conversation_history:
        trajectory_text += f"**{message['role']}**: {message['content']}\n\n"
    
    if os.environ.get("VERIFIER_API_KEY") and os.environ.get("VERIFIER_END_POINT"):
        if ground_truth:
            extra_info = {
                "acc_reward_weight": 1.0,
                "gpt_extract_answer": True,
                "extract_answer_tags": "strict",
            }
            try:
                accuracy_score, analysis = compute_score(
                    prompt=question,
                    predict_str_list=assistant_responses,
                    ground_truth=ground_truth,
                    extra_info=extra_info
                )
                trajectory_score = accuracy_score
                trajectory_analysis = analysis if analysis else "Evaluation completed using general_qa_tool."
            except Exception as e:
                print(f"Error during evaluation for {question_id}: {e}")
                accuracy_score = 0.0
                trajectory_score = 0.0
                trajectory_analysis = f"Evaluation failed: {str(e)}"
    else:
        print("VERIFIER_API_KEY or VERIFIER_END_POINT not found in environment variables. Skipping evaluation.")
        trajectory_analysis = "Evaluation skipped: VERIFIER_API_KEY or VERIFIER_END_POINT not configured."
    
    return accuracy_score, trajectory_score, trajectory_analysis, trajectory_text


def _build_result_dict(question_id, question, final_answer, ground_truth, conversation_history,
                       accuracy_score, trajectory_text, trajectory_score, trajectory_analysis):
    """
    Build result dictionary with consistent format.
    """
    return {
        "question_id": question_id,
        "prompt": question,
        "final_answer": final_answer,
        "ground_truth": ground_truth,
        "conversation_history": conversation_history,
        "accuracy_score": accuracy_score,
        "trajectory_text": trajectory_text,
        "trajectory_score": trajectory_score,
        "trajectory_analysis": trajectory_analysis,
    }


# ============================================================================
# Core Processing Functions
# ============================================================================

def _run_greedy_loop(current_node, model_caller, args, sampling_params, question_text, save_dir):
    """
    Run greedy inference loop with support for reflection.
    
    Returns:
        SearchNode: Final node after greedy inference
    """
    final_answer = None
    
    for turn in range(args.max_turns):
        # Check termination conditions
        if len(current_node.image_map) >= args.max_images:
            current_node.mark_final("Error: Reached max image limit.")
            break
        
        if current_node.current_token_count >= args.max_total_tokens - sampling_params['max_tokens']:
            current_node.mark_final("Error: Reached max token limit.")
            break
        
        # Call model
        try:
            response_text = model_caller(current_node)
        except Exception as e:
            print(f"Error calling model at turn {turn}: {e}")
            import traceback
            traceback.print_exc()
            current_node.mark_final(f"Error: Exception in model call: {str(e)}")
            # Save error to trajectory
            save_trajectory(save_dir,{
                "turn_idx": turn,
                "text_output": f"Error: Exception in model call: {str(e)}",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            break
        
        # Check if response is an error string from API
        if isinstance(response_text, str) and response_text.startswith("Error:"):
            error_msg = response_text
            print(f"Warning: {error_msg} at turn {turn}")
            current_node.mark_final(error_msg)
            # Save error to trajectory with detailed error info
            save_trajectory(save_dir,{
                "turn_idx": turn,
                "text_output": error_msg,
                "error": error_msg,
                "error_type": "api_call_failed"
            })
            break
        
        if response_text is None:
            error_msg = "Error: Empty response from model."
            print(f"Warning: {error_msg} at turn {turn}")
            current_node.mark_final(error_msg)
            # Save error to trajectory
            save_trajectory(save_dir,{
                "turn_idx": turn,
                "text_output": error_msg,
                "error": "Empty response from model"
            })
            break
        
        # Check if model_caller executed a tool call
        # model_caller now executes tool calls (like old version), so we need to check
        # if a tool call was executed and continue loop to get model's text response
        # Simple check: if the last assistant message has tool_calls, tool was executed
        tool_call_executed = False
        if current_node.api_conversation_history:
            # Find the last assistant message
            for msg in reversed(current_node.api_conversation_history):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    # If this assistant message has tool_calls, tool call was executed
                    if "tool_calls" in msg and msg["tool_calls"]:
                        tool_call_executed = True
                    break  # Only check the last assistant message
        
        # If a tool call was executed, continue loop to get model's text response based on tool results
        if tool_call_executed:
            # Tool call was executed by model_caller, continue to get model's text response
            continue
        
        # Parse response (only for non-tool-call responses)
        action, data = parse_function_call_response(response_text)
        
        if action == "answer":
            final_answer = current_node.final_answer or data
            current_node.mark_final(final_answer)
            break
        
        else:
            # Keep strict official parsing as the default. Some hosted
            # OpenAI-compatible models, however, return a normal final-text
            # message after tool use even when the official prompt asks for an
            # <answer> wrapper. This opt-in adapter lets the unchanged official
            # verifier score that visible text instead of recording a format zero.
            accept_plain_final = os.environ.get(
                "AGENTVISTA_ACCEPT_PLAIN_FINAL_TEXT", "0"
            ).lower() in ("1", "true", "yes", "on")
            candidate = response_text.strip() if isinstance(response_text, str) else ""
            if accept_plain_final and action == "text" and candidate:
                print("[Compatibility] accepting untagged final text for official verification")
                current_node.mark_final(candidate)
            else:
                current_node.mark_final("Error: Could not parse model response.")
            break
    
    # Extract final answer from node
    if current_node.is_final:
        final_answer = current_node.final_answer
    else:
        final_answer = "Error: Reached max turns without a definitive answer."
        current_node.mark_final(final_answer)
    
    return current_node


def _process_single_sample_unified(sample, args, sampling_params, trajectory_id=0):
    """
    Unified function to process a single sample.
    Supports both greedy and TTS search strategies.
    
    Args:
        sample: Data sample to process
        args: Command line arguments
        sampling_params: Sampling parameters dictionary
        trajectory_id: Trajectory identifier (for multi-rollout scenarios)
        
    Returns:
        Result dictionary with trajectory information
    """
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize sample and images (supports multiple images)
    init_result = _initialize_sample_and_image(sample, args, save_dir)
    if init_result is None:
        return None
    question_id, question, ground_truth, image_map, image_list, question_text, user_content_list = init_result
    
    # Create initial SearchNode with multiple images support
    current_system_prompt = getattr(args, '_sample_system_prompt', SYSTEM_PROMPT)
    initial_node = _create_initial_search_node(question_text, image_map, image_list, user_content_list, current_system_prompt)

    model_caller = create_model_caller(args, sampling_params, save_dir)
    final_node = _run_greedy_loop(
        initial_node, model_caller, args, sampling_params, question_text, save_dir
    )
    final_answer = final_node.final_answer
    
    # Extract responses and evaluate
    conversation_history, assistant_responses = _extract_responses_from_node(final_node)
    accuracy_score, trajectory_score, trajectory_analysis, trajectory_text = _evaluate_trajectory(
        question_text, ground_truth, conversation_history, assistant_responses, question_id
    )
    
    # Build result dict
    result = _build_result_dict(
        question_id, question_text, final_answer, ground_truth, conversation_history,
        accuracy_score, trajectory_text, trajectory_score, trajectory_analysis
    )
    
    return result


# ============================================================================
# Public API - Backward compatible aliases
# ============================================================================


def process_single_sample(sample, args, sampling_params, rollout_idx=None):
    """
    Process a single sample using either TTS or greedy search.
    
    Args:
        sample: Data sample to process
        args: Command line arguments
        sampling_params: Sampling parameters dictionary
        rollout_idx: Optional rollout index for multi-rollout mode (0-based)
        
    Returns:
        Result dictionary with trajectory information
    """
    question_id = sample.get("doc_id", sample.get("question_id", "N/A"))
    
    # Determine save directory based on rollout mode
    rollouts_per_sample = getattr(args, 'rollouts_per_sample', 1)
    if rollout_idx is not None and rollouts_per_sample > 1:
        # Multi-rollout mode: save to rollout_N subdirectory
        sample_dir = os.path.join(args.output_dir, question_id)
        save_dir = os.path.join(sample_dir, f"rollout_{rollout_idx}")
    else:
        # Single rollout mode: save directly to question_id directory
        save_dir = os.path.join(args.output_dir, question_id)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Temporarily override output_dir for processors
    original_output_dir = args.output_dir
    args.output_dir = save_dir
    
    try:
        final_result = _process_single_sample_unified(sample, args, sampling_params, trajectory_id=0)
        
        if final_result:
            # Save metrics to the rollout directory
            metrics = {
                "accuracy_score": final_result["accuracy_score"],
                "trajectory_score": final_result.get("trajectory_score"),
                "trajectory_analysis": final_result.get("trajectory_analysis"),
            }
            with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
        
        return final_result
    
    finally:
        # Restore original output_dir
        args.output_dir = original_output_dir
