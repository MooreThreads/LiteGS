from .platform import add_cmake_output_path

try:
    import litegs_fused as fused
except ImportError:
    add_cmake_output_path()
    import litegs_fused as fused
