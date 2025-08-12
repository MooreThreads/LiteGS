/*
Portions of this code are derived from the project "speedy-splat"
(https://github.com/j-alex-hanson/speedy-splat), which is based on
"gaussian-splatting" developed by Inria and the Max Planck Institute for Informatik (MPII).

Original work © Inria and MPII.  
Licensed under the Gaussian-Splatting License.  
You may use, reproduce, and distribute this work and its derivatives for
**non-commercial research and evaluation purposes only**, subject to the terms
and conditions of the Gaussian-Splatting License.

A copy of the Gaussian-Splatting License is provided in the LICENSE file.
*/


__device__ inline float2 computeEllipseIntersection(
    const float4 con_o, const float disc, const float t, const float2 p,
    const bool isY, const float coord)
{
    float p_u = isY ? p.y : p.x;
    float p_v = isY ? p.x : p.y;
    float coeff = isY ? con_o.x : con_o.z;

    float h = coord - p_u;  // h = y - p.y for y, x - p.x for x
    float sqrt_term = sqrt(disc * h * h + t * coeff);

    return {
      (-con_o.y * h - sqrt_term) / coeff + p_v,
      (-con_o.y * h + sqrt_term) / coeff + p_v
    };
}

template<int TilesizeY,int TilesizeX>
__device__ inline uint32_t processTiles(
    const float4 con_o, const float disc, const float t, const float2 p,
    float2 bbox_min, float2 bbox_max,
    float2 bbox_argmin, float2 bbox_argmax,
    int2 rect_min, int2 rect_max,
    const dim3 grid, const bool isY,
    uint32_t idx, uint32_t off,// float depth,
    int* gaussian_keys_unsorted,
    int* gaussian_values_unsorted
)
{

    // ---- AccuTile Code ---- //

    // Set variables based on the isY flag
    float BLOCK_U = isY ? TilesizeY : TilesizeX;
    float BLOCK_V = isY ? TilesizeX : TilesizeY;

    if (isY) {
        rect_min = { rect_min.y, rect_min.x };
        rect_max = { rect_max.y, rect_max.x };

        bbox_min = { bbox_min.y, bbox_min.x };
        bbox_max = { bbox_max.y, bbox_max.x };

        bbox_argmin = { bbox_argmin.y, bbox_argmin.x };
        bbox_argmax = { bbox_argmax.y, bbox_argmax.x };
    }

    uint32_t tiles_count = 0;
    float2 intersect_min_line, intersect_max_line;
    float ellipse_min, ellipse_max;
    float min_line, max_line;

    // Initialize max line
    // Just need the min to be >= all points on the ellipse
    // and  max to be <= all points on the ellipse
    intersect_max_line = { bbox_max.y, bbox_min.y };

    min_line = rect_min.x * BLOCK_U;
    // Initialize min line intersections.
    if (bbox_min.x <= min_line) {
        // Boundary case
        intersect_min_line = computeEllipseIntersection(
            con_o, disc, t, p, isY, rect_min.x * BLOCK_U);

    }
    else {
        // Same as max line
        intersect_min_line = intersect_max_line;
    }


    // Loop over either y slices or x slices based on the `isY` flag.
    for (int u = rect_min.x; u < rect_max.x; ++u)
    {
        // Starting from the bottom or left, we will only need to compute
        // intersections at the next line.
        max_line = min_line + BLOCK_U;
        if (max_line <= bbox_max.x) {
            intersect_max_line = computeEllipseIntersection(
                con_o, disc, t, p, isY, max_line);
        }

        // If the bbox min is in this slice, then it is the minimum
        // ellipse point in this slice. Otherwise, the minimum ellipse
        // point will be the minimum of the intersections of the min/max lines.
        if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
            ellipse_min = bbox_min.y;
        }
        else {
            ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
        }

        // If the bbox max is in this slice, then it is the maximum
        // ellipse point in this slice. Otherwise, the maximum ellipse
        // point will be the maximum of the intersections of the min/max lines.
        if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
            ellipse_max = bbox_max.y;
        }
        else {
            ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
        }

        // Convert ellipse_min/ellipse_max to tiles touched
        // First map back to tile coordinates, then subtract.
        int min_tile_v = max(rect_min.y,
            min(rect_max.y, (int)(ellipse_min / BLOCK_V))
        );
        int max_tile_v = min(rect_max.y,
            max(rect_min.y, (int)(ellipse_max / BLOCK_V + 1))
        );

        tiles_count += max_tile_v - min_tile_v;
        // Only update keys array if it exists.
        if (gaussian_keys_unsorted != nullptr) {
            // Loop over tiles and add to keys array
            for (int v = min_tile_v; v < max_tile_v; v++)
            {
                // For each tile that the Gaussian overlaps, emit a
                // key/value pair. The key is |  tile ID ,
                // and the value is the ID of the Gaussian. Sorting the values
                // with this key yields Gaussian IDs in a list, such that they
                // are first sorted by tile and then by depth.
                uint32_t key = isY ? (u * grid.x + v) : (v * grid.x + u);
                gaussian_keys_unsorted[off] = key+1;//offset 1 , tile_id 0 means invald
                gaussian_values_unsorted[off] = idx;
                off++;
            }
        }
        // Max line of this tile slice will be min lin of next tile slice
        intersect_min_line = intersect_max_line;
        min_line = max_line;
    }
    return tiles_count;
}