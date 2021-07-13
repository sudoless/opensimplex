// Package opensimplex is a Go implementation of Kurt Spencer's patent-free alternative to Perlin and Simplex noise.
//
// Given a seed, it generates smoothly-changing deterministic random values in
// 2, 3 or 4 dimensions. It's commonly used for procedurally generated images,
// geometry, or other randomly-influenced applications that require a random
// gradient.
//
// For more information on OpenSimplex noise, read more from the creator of the
// algorithm: http://uniblock.tumblr.com/post/97868843242/noise
package opensimplex

/**
 * OpenSimplex Noise in Go.
 * algorithm by Kurt Spencer
 * ported by Owen Raccuglia
 *
 * Based on Java v1.1 (October 5, 2014)
 */

// Noise is a seeded 64-bit noise instance
type Noise interface {
	Eval2(x, y float64) float64
	Eval3(x, y, z float64) float64
	Eval4(x, y, z, w float64) float64
}

// Noise32 is a seeded 32-bit noise instance
type Noise32 interface {
	Eval2(x, y float32) float32
	Eval3(x, y, z float32) float32
	Eval4(x, y, z, w float32) float32
}

// New constructs a Noise instance with a 64-bit seed.
func New(seed int64) Noise {
	s := &noise{}

	source := make([]int16, 256)
	for i := range source {
		source[i] = int16(i)
	}

	gradientLenOver3 := int16(len(gradients3D)) / 3

	seed = seed*6364136223846793005 + 1442695040888963407
	seed = seed*6364136223846793005 + 1442695040888963407
	seed = seed*6364136223846793005 + 1442695040888963407
	for i := int32(255); i >= 0; i-- {
		seed = seed*6364136223846793005 + 1442695040888963407
		r := int32((seed + 31) % int64(i+1))
		if r < 0 {
			r += i + 1
		}

		s.perm[i] = source[r]
		s.permGradIndex3D[i] = (s.perm[i] % gradientLenOver3) * 3
		source[r] = source[i]
	}

	return s
}

// New32 constructs a Noise32 instance with a 64-bit seed.
func New32(seed int64) Noise32 {
	return &cast32Noise{base: New(seed)}
}

// NewNormalized constructs a normalized Noise instance with a 64-bit seed. Eval methods will
// return values in [0, 1).
func NewNormalized(seed int64) Noise {
	return &normNoise{base: New(seed)}
}

// NewNormalized32 constructs a normalized Noise32 instance with a 64-bit seed. Eval methods will
// return values in [0, 1).
func NewNormalized32(seed int64) Noise32 {
	return &normNoise32{base: New(seed)}
}
