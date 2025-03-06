from .. import arguments

def get_default()->tuple[arguments.OptimizationParams,arguments.PipelineParams]:
    return arguments.OptimizationParams(),arguments.PipelineParams()