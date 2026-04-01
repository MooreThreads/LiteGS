from .platform import add_cmake_output_path


def load_fused_backend():
    try:
        import litegs_fused
    except ImportError:
        add_cmake_output_path()
        import litegs_fused
    return litegs_fused
