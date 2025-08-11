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