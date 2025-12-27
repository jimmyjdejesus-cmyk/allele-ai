import os


def test_env_override_agent_model():
    # Set env var to change AGENT model name
    os.environ["AGENT__MODEL_NAME"] = "test-model-env"

    # Reload config module so runtime settings are recreated from env
    import importlib

    import phylogenic.config as _config

    importlib.reload(_config)

    from phylogenic.config import settings as module_settings

    assert module_settings.agent.model_name == "test-model-env"

    # Cleanup
    del os.environ["AGENT__MODEL_NAME"]


def test_env_override_evolution_immutable_and_hpc_flags():
    os.environ["EVOLUTION__IMMUTABLE_EVOLUTION"] = "true"
    os.environ["EVOLUTION__HPC_MODE"] = "false"

    import importlib

    import phylogenic.config as _config

    importlib.reload(_config)

    from phylogenic.config import settings as module_settings

    assert module_settings.evolution.immutable_evolution is True
    assert module_settings.evolution.hpc_mode is False

    del os.environ["EVOLUTION__IMMUTABLE_EVOLUTION"]
    del os.environ["EVOLUTION__HPC_MODE"]


