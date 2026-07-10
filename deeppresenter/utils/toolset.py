from deeppresenter.utils.typings import ToolSet


def resolve_toolset(
    toolset: ToolSet,
    tools_dict: dict[str, dict],
    server_tools: dict[str, list[str]],
) -> list[dict]:
    """Filter a tool registry (name->spec, server->[names]) down to the specs a
    role's ToolSet allows. Shared by the old Agent._setup_toolset and the new
    LangGraph-based engine so both stay in lockstep."""
    if toolset.include_tool_servers == "all":
        servers = list(server_tools.keys())
    else:
        servers = toolset.include_tool_servers

    tools: list[dict] = []
    for server in servers:
        if server in toolset.exclude_tool_servers:
            continue
        for tool_name in server_tools.get(server, []):
            if tool_name not in toolset.exclude_tools:
                tools.append(tools_dict[tool_name])

    for tool_name in toolset.include_tools:
        spec = tools_dict.get(tool_name)
        if spec and spec not in tools:
            tools.append(spec)

    return tools
