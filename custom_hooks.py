from litellm.integrations.custom_logger import CustomLogger


class PhiToolChoiceFix(CustomLogger):
    """Strip tool_choice from requests to Phi models.

    Azure Phi endpoints accept the 'tools' array but reject the 'tool_choice'
    parameter, causing a validation error. We remove tool_choice while keeping
    any tool definitions so the model can still see them in context.
    """

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        model = data.get("model", "").lower()
        if "phi" in model and "tool_choice" in data:
            del data["tool_choice"]
        return data


# Module-level instance required for LiteLLM proxy callback registration
phi_tool_choice_fix = PhiToolChoiceFix()
