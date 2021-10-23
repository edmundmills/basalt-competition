import contexts.minerl.environment as minerl_env


def start_env(config, debug_env=False):
    context = config.context.name
    if context == 'MineRL':
        return minerl_env.start_env(config, debug_env)
