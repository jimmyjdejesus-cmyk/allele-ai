import os


def test_env_override_agent_model():
    # Set env var to change AGENT model name
    os.environ['AGENT__MODEL_NAME'] = 'test-model-env'

    # Recreate settings to pick up environment change
    from allele.config import AlleleSettings
    config = AlleleSettings()

    assert config.agent.model_name == 'test-model-env'

    # Cleanup
    del os.environ['AGENT__MODEL_NAME']


def test_env_override_evolution_immutable_and_hpc_flags():
    os.environ['EVOLUTION__IMMUTABLE_EVOLUTION'] = 'true'
    os.environ['EVOLUTION__HPC_MODE'] = 'false'

    from allele.config import AlleleSettings
    config = AlleleSettings()

    assert config.evolution.immutable_evolution is True
    assert config.evolution.hpc_mode is False

    del os.environ['EVOLUTION__IMMUTABLE_EVOLUTION']
    del os.environ['EVOLUTION__HPC_MODE']
