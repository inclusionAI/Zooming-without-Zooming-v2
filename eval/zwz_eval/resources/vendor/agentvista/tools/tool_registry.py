"""
Tool registry that manages all available tools.
Provides a registration decorator and query utilities.
"""

TOOL_REGISTRY = {}


def register_tool(name: str = None):
    """
    Decorator for registering a tool class.

    Args:
        name: Tool name. Falls back to the class's ``name`` attribute if not given.

    Usage:
        @register_tool("my_tool")
        class MyTool(BaseTool):
            ...
    """
    def decorator(cls):
        tool_name = name or cls.name
        if tool_name in TOOL_REGISTRY:
            print(f"Warning: Tool '{tool_name}' already exists, overwriting")
        TOOL_REGISTRY[tool_name] = cls
        cls.name = tool_name
        return cls
    return decorator


def get_tool(name: str):
    """
    Retrieve a tool class (not instantiated) by name.

    Args:
        name: Tool name.

    Returns:
        The tool class.

    Raises:
        ValueError: If the tool is not registered.
    """
    if name not in TOOL_REGISTRY:
        available = list_tools()
        raise ValueError(
            f"Tool '{name}' not found in registry. "
            f"Available tools: {', '.join(available)}"
        )
    return TOOL_REGISTRY[name]


def list_tools():
    """
    List all registered tool names.

    Returns:
        A list of tool name strings.
    """
    return list(TOOL_REGISTRY.keys())


def get_tool_info(name: str):
    """
    Get detailed information about a registered tool.

    Args:
        name: Tool name.

    Returns:
        A dictionary containing name, description, and parameters.
    """
    tool_cls = get_tool(name)
    return {
        "name": tool_cls.name,
        "description": tool_cls.description,
        "parameters": tool_cls.parameters
    }
