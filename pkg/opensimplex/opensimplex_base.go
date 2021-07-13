package opensimplex

import "math"

// Vanilla opensimplex implementation, matching Kurt Spencer's Java
// reference implementation as exactly as possible.

// A seeded Noise instance. Reusing a Noise instance (rather than recreating it
// from a known seed) will save some calculation time.
type noise struct {
	perm            [256]int16
	permGradIndex3D [256]int16
}

// Eval2 returns a random noise value in two dimensions. Repeated calls with the same
// x/y inputs will have the same output.
func (s *noise) Eval2(x, y float64) float64 {
	// Place input coordinates onto grid.
	stretchOffset := (x + y) * stretchConstant2D
	xs := x + stretchOffset
	ys := y + stretchOffset

	// Floor to get grid coordinates of rhombus (stretched square) super-cell origin.
	xsb := int32(math.Floor(xs))
	ysb := int32(math.Floor(ys))

	// Skew out to get actual coordinates of rhombus origin. We'll need these later.
	squishOffset := float64(xsb+ysb) * squishConstant2D
	xb := float64(xsb) + squishOffset
	yb := float64(ysb) + squishOffset

	// Compute grid coordinates relative to rhombus origin.
	xins := xs - float64(xsb)
	yins := ys - float64(ysb)

	// Sum those together to get a value that determines which region we're in.
	inSum := xins + yins

	// Positions relative to origin point.
	dx0 := x - xb
	dy0 := y - yb

	// We'll be defining these inside the next block and using them afterwards.
	var dxExt, dyExt float64
	var xsvExt, ysvExt int32

	value := float64(0)

	// Contribution (1,0)
	dx1 := dx0 - 1 - squishConstant2D
	dy1 := dy0 - 0 - squishConstant2D
	attn1 := 2 - dx1*dx1 - dy1*dy1
	if attn1 > 0 {
		attn1 *= attn1
		value += attn1 * attn1 * s.extrapolate2(xsb+1, ysb+0, dx1, dy1)
	}

	// Contribution (0,1)
	dx2 := dx0 - 0 - squishConstant2D
	dy2 := dy0 - 1 - squishConstant2D
	attn2 := 2 - dx2*dx2 - dy2*dy2
	if attn2 > 0 {
		attn2 *= attn2
		value += attn2 * attn2 * s.extrapolate2(xsb+0, ysb+1, dx2, dy2)
	}

	if inSum <= 1 { // We're inside the triangle (2-Simplex) at (0,0)
		zins := 1 - inSum
		if zins > xins || zins > yins { // (0,0) is one of the closest two triangular vertices
			if xins > yins {
				xsvExt = xsb + 1
				ysvExt = ysb - 1
				dxExt = dx0 - 1
				dyExt = dy0 + 1
			} else {
				xsvExt = xsb - 1
				ysvExt = ysb + 1
				dxExt = dx0 + 1
				dyExt = dy0 - 1
			}
		} else { // (1,0) and (0,1) are the closest two vertices.
			xsvExt = xsb + 1
			ysvExt = ysb + 1
			dxExt = dx0 - 1 - 2*squishConstant2D
			dyExt = dy0 - 1 - 2*squishConstant2D
		}
	} else { // We're inside the triangle (2-Simplex) at (1,1)
		zins := 2 - inSum
		if zins < xins || zins < yins { // (0,0) is one of the closest two triangular vertices
			if xins > yins {
				xsvExt = xsb + 2
				ysvExt = ysb + 0
				dxExt = dx0 - 2 - 2*squishConstant2D
				dyExt = dy0 + 0 - 2*squishConstant2D
			} else {
				xsvExt = xsb + 0
				ysvExt = ysb + 2
				dxExt = dx0 + 0 - 2*squishConstant2D
				dyExt = dy0 - 2 - 2*squishConstant2D
			}
		} else { // (1,0) and (0,1) are the closest two vertices.
			dxExt = dx0
			dyExt = dy0
			xsvExt = xsb
			ysvExt = ysb
		}
		xsb += 1
		ysb += 1
		dx0 = dx0 - 1 - 2*squishConstant2D
		dy0 = dy0 - 1 - 2*squishConstant2D
	}

	// Contribution (0,0) or (1,1)
	attn0 := 2 - dx0*dx0 - dy0*dy0
	if attn0 > 0 {
		attn0 *= attn0
		value += attn0 * attn0 * s.extrapolate2(xsb, ysb, dx0, dy0)
	}

	// Extra Vertex
	attnExt := 2 - dxExt*dxExt - dyExt*dyExt
	if attnExt > 0 {
		attnExt *= attnExt
		value += attnExt * attnExt * s.extrapolate2(xsvExt, ysvExt, dxExt, dyExt)
	}

	return value / normConstant2D
}

// Eval3 returns a random noise value in three dimensions.
func (s *noise) Eval3(x, y, z float64) float64 {
	// Place input coordinates on simplectic honeycomb.
	stretchOffset := (x + y + z) * stretchConstant3D
	xs := x + stretchOffset
	ys := y + stretchOffset
	zs := z + stretchOffset

	// Floor to get simplectic honeycomb coordinates of rhombohedron (stretched cube) super-cell origin.
	xsb := int32(math.Floor(xs))
	ysb := int32(math.Floor(ys))
	zsb := int32(math.Floor(zs))

	// Skew out to get actual coordinates of rhombohedron origin. We'll need these later.
	squishOffset := float64(xsb+ysb+zsb) * squishConstant3D
	xb := float64(xsb) + squishOffset
	yb := float64(ysb) + squishOffset
	zb := float64(zsb) + squishOffset

	// Compute simplectic honeycomb coordinates relative to rhombohedral origin.
	xins := xs - float64(xsb)
	yins := ys - float64(ysb)
	zins := zs - float64(zsb)

	// Sum those together to get a value that determines which region we're in.
	inSum := xins + yins + zins

	// Positions relative to origin point.
	dx0 := x - xb
	dy0 := y - yb
	dz0 := z - zb

	// We'll be defining these inside the next block and using them afterwards.
	var dxExt0, dyExt0, dzExt0 float64
	var dxExt1, dyExt1, dzExt1 float64
	var xsvExt0, ysvExt0, zsvExt0 int32
	var xsvExt1, ysvExt1, zsvExt1 int32

	value := float64(0)
	if inSum <= 1 { // We're inside the tetrahedron (3-Simplex) at (0,0,0)

		// Determine which two of (0,0,1), (0,1,0), (1,0,0) are closest.
		aPoint := byte(0x01)
		bPoint := byte(0x02)
		aScore := xins
		bScore := yins
		if aScore >= bScore && zins > bScore {
			bScore = zins
			bPoint = 0x04
		} else if aScore < bScore && zins > aScore {
			aScore = zins
			aPoint = 0x04
		}

		// Now we determine the two lattice points not part of the tetrahedron that may contribute.
		// This depends on the closest two tetrahedral vertices, including (0,0,0)
		wins := 1 - inSum
		if wins > aScore || wins > bScore { // (0,0,0) is one of the closest two tetrahedral vertices.
			var c byte // Our other closest vertex is the closest out of a and b.
			if bScore > aScore {
				c = bPoint
			} else {
				c = aPoint
			}

			if (c & 0x01) == 0 {
				xsvExt0 = xsb - 1
				xsvExt1 = xsb
				dxExt0 = dx0 + 1
				dxExt1 = dx0
			} else {
				xsvExt1 = xsb + 1
				xsvExt0 = xsvExt1
				dxExt1 = dx0 - 1
				dxExt0 = dxExt1
			}

			if (c & 0x02) == 0 {
				ysvExt1 = ysb
				ysvExt0 = ysvExt1
				dyExt1 = dy0
				dyExt0 = dyExt1
				if (c & 0x01) == 0 {
					ysvExt1 -= 1
					dyExt1 += 1
				} else {
					ysvExt0 -= 1
					dyExt0 += 1
				}
			} else {
				ysvExt1 = ysb + 1
				ysvExt0 = ysvExt1
				dyExt1 = dy0 - 1
				dyExt0 = dyExt1
			}

			if (c & 0x04) == 0 {
				zsvExt0 = zsb
				zsvExt1 = zsb - 1
				dzExt0 = dz0
				dzExt1 = dz0 + 1
			} else {
				zsvExt1 = zsb + 1
				zsvExt0 = zsvExt1
				dzExt1 = dz0 - 1
				dzExt0 = dzExt1
			}
		} else { // (0,0,0) is not one of the closest two tetrahedral vertices.
			c := aPoint | bPoint // Our two extra vertices are determined by the closest two.

			if (c & 0x01) == 0 {
				xsvExt0 = xsb
				xsvExt1 = xsb - 1
				dxExt0 = dx0 - 2*squishConstant3D
				dxExt1 = dx0 + 1 - squishConstant3D
			} else {
				xsvExt1 = xsb + 1
				xsvExt0 = xsvExt1
				dxExt0 = dx0 - 1 - 2*squishConstant3D
				dxExt1 = dx0 - 1 - squishConstant3D
			}

			if (c & 0x02) == 0 {
				ysvExt0 = ysb
				ysvExt1 = ysb - 1
				dyExt0 = dy0 - 2*squishConstant3D
				dyExt1 = dy0 + 1 - squishConstant3D
			} else {
				ysvExt1 = ysb + 1
				ysvExt0 = ysvExt1
				dyExt0 = dy0 - 1 - 2*squishConstant3D
				dyExt1 = dy0 - 1 - squishConstant3D
			}

			if (c & 0x04) == 0 {
				zsvExt0 = zsb
				zsvExt1 = zsb - 1
				dzExt0 = dz0 - 2*squishConstant3D
				dzExt1 = dz0 + 1 - squishConstant3D
			} else {
				zsvExt1 = zsb + 1
				zsvExt0 = zsvExt1
				dzExt0 = dz0 - 1 - 2*squishConstant3D
				dzExt1 = dz0 - 1 - squishConstant3D
			}
		}

		// Contribution (0,0,0)
		attn0 := 2 - dx0*dx0 - dy0*dy0 - dz0*dz0
		if attn0 > 0 {
			attn0 *= attn0
			value += attn0 * attn0 * s.extrapolate3(xsb+0, ysb+0, zsb+0, dx0, dy0, dz0)
		}

		// Contribution (1,0,0)
		dx1 := dx0 - 1 - squishConstant3D
		dy1 := dy0 - 0 - squishConstant3D
		dz1 := dz0 - 0 - squishConstant3D
		attn1 := 2 - dx1*dx1 - dy1*dy1 - dz1*dz1
		if attn1 > 0 {
			attn1 *= attn1
			value += attn1 * attn1 * s.extrapolate3(xsb+1, ysb+0, zsb+0, dx1, dy1, dz1)
		}

		// Contribution (0,1,0)
		dx2 := dx0 - 0 - squishConstant3D
		dy2 := dy0 - 1 - squishConstant3D
		dz2 := dz1
		attn2 := 2 - dx2*dx2 - dy2*dy2 - dz2*dz2
		if attn2 > 0 {
			attn2 *= attn2
			value += attn2 * attn2 * s.extrapolate3(xsb+0, ysb+1, zsb+0, dx2, dy2, dz2)
		}

		// Contribution (0,0,1)
		dx3 := dx2
		dy3 := dy1
		dz3 := dz0 - 1 - squishConstant3D
		attn3 := 2 - dx3*dx3 - dy3*dy3 - dz3*dz3
		if attn3 > 0 {
			attn3 *= attn3
			value += attn3 * attn3 * s.extrapolate3(xsb+0, ysb+0, zsb+1, dx3, dy3, dz3)
		}
	} else if inSum >= 2 { // We're inside the tetrahedron (3-Simplex) at (1,1,1)

		// Determine which two tetrahedral vertices are the closest, out of (1,1,0), (1,0,1), (0,1,1) but not (1,1,1).
		aPoint := byte(0x06)
		aScore := xins
		bPoint := byte(0x05)
		bScore := yins
		if aScore <= bScore && zins < bScore {
			bScore = zins
			bPoint = 0x03
		} else if aScore > bScore && zins < aScore {
			aScore = zins
			aPoint = 0x03
		}

		// Now we determine the two lattice points not part of the tetrahedron that may contribute.
		// This depends on the closest two tetrahedral vertices, including (1,1,1)
		wins := 3 - inSum
		if wins < aScore || wins < bScore { // (1,1,1) is one of the closest two tetrahedral vertices.
			var c byte // Our other closest vertex is the closest out of a and b.
			if bScore < aScore {
				c = bPoint
			} else {
				c = aPoint
			}

			if (c & 0x01) != 0 {
				xsvExt0 = xsb + 2
				xsvExt1 = xsb + 1
				dxExt0 = dx0 - 2 - 3*squishConstant3D
				dxExt1 = dx0 - 1 - 3*squishConstant3D
			} else {
				xsvExt1 = xsb
				xsvExt0 = xsvExt1
				dxExt1 = dx0 - 3*squishConstant3D
				dxExt0 = dxExt1
			}

			if (c & 0x02) != 0 {
				ysvExt1 = ysb + 1
				ysvExt0 = ysvExt1
				dyExt1 = dy0 - 1 - 3*squishConstant3D
				dyExt0 = dyExt1
				if (c & 0x01) != 0 {
					ysvExt1 += 1
					dyExt1 -= 1
				} else {
					ysvExt0 += 1
					dyExt0 -= 1
				}
			} else {
				ysvExt1 = ysb
				ysvExt0 = ysvExt1
				dyExt1 = dy0 - 3*squishConstant3D
				dyExt0 = dyExt1
			}

			if (c & 0x04) != 0 {
				zsvExt0 = zsb + 1
				zsvExt1 = zsb + 2
				dzExt0 = dz0 - 1 - 3*squishConstant3D
				dzExt1 = dz0 - 2 - 3*squishConstant3D
			} else {
				zsvExt1 = zsb
				zsvExt0 = zsvExt1
				dzExt1 = dz0 - 3*squishConstant3D
				dzExt0 = dzExt1
			}
		} else { // (1,1,1) is not one of the closest two tetrahedral vertices.
			c := aPoint & bPoint // Our two extra vertices are determined by the closest two.

			if (c & 0x01) != 0 {
				xsvExt0 = xsb + 1
				xsvExt1 = xsb + 2
				dxExt0 = dx0 - 1 - squishConstant3D
				dxExt1 = dx0 - 2 - 2*squishConstant3D
			} else {
				xsvExt1 = xsb
				xsvExt0 = xsvExt1
				dxExt0 = dx0 - squishConstant3D
				dxExt1 = dx0 - 2*squishConstant3D
			}

			if (c & 0x02) != 0 {
				ysvExt0 = ysb + 1
				ysvExt1 = ysb + 2
				dyExt0 = dy0 - 1 - squishConstant3D
				dyExt1 = dy0 - 2 - 2*squishConstant3D
			} else {
				ysvExt1 = ysb
				ysvExt0 = ysvExt1
				dyExt0 = dy0 - squishConstant3D
				dyExt1 = dy0 - 2*squishConstant3D
			}

			if (c & 0x04) != 0 {
				zsvExt0 = zsb + 1
				zsvExt1 = zsb + 2
				dzExt0 = dz0 - 1 - squishConstant3D
				dzExt1 = dz0 - 2 - 2*squishConstant3D
			} else {
				zsvExt1 = zsb
				zsvExt0 = zsvExt1
				dzExt0 = dz0 - squishConstant3D
				dzExt1 = dz0 - 2*squishConstant3D
			}
		}

		// Contribution (1,1,0)
		dx3 := dx0 - 1 - 2*squishConstant3D
		dy3 := dy0 - 1 - 2*squishConstant3D
		dz3 := dz0 - 0 - 2*squishConstant3D
		attn3 := 2 - dx3*dx3 - dy3*dy3 - dz3*dz3
		if attn3 > 0 {
			attn3 *= attn3
			value += attn3 * attn3 * s.extrapolate3(xsb+1, ysb+1, zsb+0, dx3, dy3, dz3)
		}

		// Contribution (1,0,1)
		dx2 := dx3
		dy2 := dy0 - 0 - 2*squishConstant3D
		dz2 := dz0 - 1 - 2*squishConstant3D
		attn2 := 2 - dx2*dx2 - dy2*dy2 - dz2*dz2
		if attn2 > 0 {
			attn2 *= attn2
			value += attn2 * attn2 * s.extrapolate3(xsb+1, ysb+0, zsb+1, dx2, dy2, dz2)
		}

		// Contribution (0,1,1)
		dx1 := dx0 - 0 - 2*squishConstant3D
		dy1 := dy3
		dz1 := dz2
		attn1 := 2 - dx1*dx1 - dy1*dy1 - dz1*dz1
		if attn1 > 0 {
			attn1 *= attn1
			value += attn1 * attn1 * s.extrapolate3(xsb+0, ysb+1, zsb+1, dx1, dy1, dz1)
		}

		// Contribution (1,1,1)
		dx0 = dx0 - 1 - 3*squishConstant3D
		dy0 = dy0 - 1 - 3*squishConstant3D
		dz0 = dz0 - 1 - 3*squishConstant3D
		attn0 := 2 - dx0*dx0 - dy0*dy0 - dz0*dz0
		if attn0 > 0 {
			attn0 *= attn0
			value += attn0 * attn0 * s.extrapolate3(xsb+1, ysb+1, zsb+1, dx0, dy0, dz0)
		}
	} else { // We're inside the octahedron (Rectified 3-Simplex) in between.
		var aScore, bScore float64
		var aPoint, bPoint byte
		var aIsFurtherSide, bIsFurtherSide bool

		// Decide between point (0,0,1) and (1,1,0) as closest
		p1 := xins + yins
		if p1 > 1 {
			aScore = p1 - 1
			aPoint = 0x03
			aIsFurtherSide = true
		} else {
			aScore = 1 - p1
			aPoint = 0x04
			aIsFurtherSide = false
		}

		// Decide between point (0,1,0) and (1,0,1) as closest
		p2 := xins + zins
		if p2 > 1 {
			bScore = p2 - 1
			bPoint = 0x05
			bIsFurtherSide = true
		} else {
			bScore = 1 - p2
			bPoint = 0x02
			bIsFurtherSide = false
		}

		// The closest out of the two (1,0,0) and (0,1,1) will replace the furthest out of the two decided above, if closer.
		p3 := yins + zins
		if p3 > 1 {
			score := p3 - 1
			if aScore <= bScore && aScore < score {
				aPoint = 0x06
				aIsFurtherSide = true
			} else if aScore > bScore && bScore < score {
				bPoint = 0x06
				bIsFurtherSide = true
			}
		} else {
			score := 1 - p3
			if aScore <= bScore && aScore < score {
				aPoint = 0x01
				aIsFurtherSide = false
			} else if aScore > bScore && bScore < score {
				bPoint = 0x01
				bIsFurtherSide = false
			}
		}

		// Where each of the two closest points are determines how the extra two vertices are calculated.
		if aIsFurtherSide == bIsFurtherSide {
			if aIsFurtherSide { // Both closest points on (1,1,1) side

				// One of the two extra points is (1,1,1)
				dxExt0 = dx0 - 1 - 3*squishConstant3D
				dyExt0 = dy0 - 1 - 3*squishConstant3D
				dzExt0 = dz0 - 1 - 3*squishConstant3D
				xsvExt0 = xsb + 1
				ysvExt0 = ysb + 1
				zsvExt0 = zsb + 1

				// Other extra point is based on the shared axis.
				c := aPoint & bPoint
				if (c & 0x01) != 0 {
					dxExt1 = dx0 - 2 - 2*squishConstant3D
					dyExt1 = dy0 - 2*squishConstant3D
					dzExt1 = dz0 - 2*squishConstant3D
					xsvExt1 = xsb + 2
					ysvExt1 = ysb
					zsvExt1 = zsb
				} else if (c & 0x02) != 0 {
					dxExt1 = dx0 - 2*squishConstant3D
					dyExt1 = dy0 - 2 - 2*squishConstant3D
					dzExt1 = dz0 - 2*squishConstant3D
					xsvExt1 = xsb
					ysvExt1 = ysb + 2
					zsvExt1 = zsb
				} else {
					dxExt1 = dx0 - 2*squishConstant3D
					dyExt1 = dy0 - 2*squishConstant3D
					dzExt1 = dz0 - 2 - 2*squishConstant3D
					xsvExt1 = xsb
					ysvExt1 = ysb
					zsvExt1 = zsb + 2
				}
			} else { // Both closest points on (0,0,0) side

				// One of the two extra points is (0,0,0)
				dxExt0 = dx0
				dyExt0 = dy0
				dzExt0 = dz0
				xsvExt0 = xsb
				ysvExt0 = ysb
				zsvExt0 = zsb

				// Other extra point is based on the omitted axis.
				c := aPoint | bPoint
				if (c & 0x01) == 0 {
					dxExt1 = dx0 + 1 - squishConstant3D
					dyExt1 = dy0 - 1 - squishConstant3D
					dzExt1 = dz0 - 1 - squishConstant3D
					xsvExt1 = xsb - 1
					ysvExt1 = ysb + 1
					zsvExt1 = zsb + 1
				} else if (c & 0x02) == 0 {
					dxExt1 = dx0 - 1 - squishConstant3D
					dyExt1 = dy0 + 1 - squishConstant3D
					dzExt1 = dz0 - 1 - squishConstant3D
					xsvExt1 = xsb + 1
					ysvExt1 = ysb - 1
					zsvExt1 = zsb + 1
				} else {
					dxExt1 = dx0 - 1 - squishConstant3D
					dyExt1 = dy0 - 1 - squishConstant3D
					dzExt1 = dz0 + 1 - squishConstant3D
					xsvExt1 = xsb + 1
					ysvExt1 = ysb + 1
					zsvExt1 = zsb - 1
				}
			}
		} else { // One point on (0,0,0) side, one point on (1,1,1) side
			var c1, c2 byte
			if aIsFurtherSide {
				c1 = aPoint
				c2 = bPoint
			} else {
				c1 = bPoint
				c2 = aPoint
			}

			// One contribution is a permutation of (1,1,-1)
			if (c1 & 0x01) == 0 {
				dxExt0 = dx0 + 1 - squishConstant3D
				dyExt0 = dy0 - 1 - squishConstant3D
				dzExt0 = dz0 - 1 - squishConstant3D
				xsvExt0 = xsb - 1
				ysvExt0 = ysb + 1
				zsvExt0 = zsb + 1
			} else if (c1 & 0x02) == 0 {
				dxExt0 = dx0 - 1 - squishConstant3D
				dyExt0 = dy0 + 1 - squishConstant3D
				dzExt0 = dz0 - 1 - squishConstant3D
				xsvExt0 = xsb + 1
				ysvExt0 = ysb - 1
				zsvExt0 = zsb + 1
			} else {
				dxExt0 = dx0 - 1 - squishConstant3D
				dyExt0 = dy0 - 1 - squishConstant3D
				dzExt0 = dz0 + 1 - squishConstant3D
				xsvExt0 = xsb + 1
				ysvExt0 = ysb + 1
				zsvExt0 = zsb - 1
			}

			// One contribution is a permutation of (0,0,2)
			dxExt1 = dx0 - 2*squishConstant3D
			dyExt1 = dy0 - 2*squishConstant3D
			dzExt1 = dz0 - 2*squishConstant3D
			xsvExt1 = xsb
			ysvExt1 = ysb
			zsvExt1 = zsb
			if (c2 & 0x01) != 0 {
				dxExt1 -= 2
				xsvExt1 += 2
			} else if (c2 & 0x02) != 0 {
				dyExt1 -= 2
				ysvExt1 += 2
			} else {
				dzExt1 -= 2
				zsvExt1 += 2
			}
		}

		// Contribution (1,0,0)
		dx1 := dx0 - 1 - squishConstant3D
		dy1 := dy0 - 0 - squishConstant3D
		dz1 := dz0 - 0 - squishConstant3D
		attn1 := 2 - dx1*dx1 - dy1*dy1 - dz1*dz1
		if attn1 > 0 {
			attn1 *= attn1
			value += attn1 * attn1 * s.extrapolate3(xsb+1, ysb+0, zsb+0, dx1, dy1, dz1)
		}

		// Contribution (0,1,0)
		dx2 := dx0 - 0 - squishConstant3D
		dy2 := dy0 - 1 - squishConstant3D
		dz2 := dz1
		attn2 := 2 - dx2*dx2 - dy2*dy2 - dz2*dz2
		if attn2 > 0 {
			attn2 *= attn2
			value += attn2 * attn2 * s.extrapolate3(xsb+0, ysb+1, zsb+0, dx2, dy2, dz2)
		}

		// Contribution (0,0,1)
		dx3 := dx2
		dy3 := dy1
		dz3 := dz0 - 1 - squishConstant3D
		attn3 := 2 - dx3*dx3 - dy3*dy3 - dz3*dz3
		if attn3 > 0 {
			attn3 *= attn3
			value += attn3 * attn3 * s.extrapolate3(xsb+0, ysb+0, zsb+1, dx3, dy3, dz3)
		}

		// Contribution (1,1,0)
		dx4 := dx0 - 1 - 2*squishConstant3D
		dy4 := dy0 - 1 - 2*squishConstant3D
		dz4 := dz0 - 0 - 2*squishConstant3D
		attn4 := 2 - dx4*dx4 - dy4*dy4 - dz4*dz4
		if attn4 > 0 {
			attn4 *= attn4
			value += attn4 * attn4 * s.extrapolate3(xsb+1, ysb+1, zsb+0, dx4, dy4, dz4)
		}

		// Contribution (1,0,1)
		dx5 := dx4
		dy5 := dy0 - 0 - 2*squishConstant3D
		dz5 := dz0 - 1 - 2*squishConstant3D
		attn5 := 2 - dx5*dx5 - dy5*dy5 - dz5*dz5
		if attn5 > 0 {
			attn5 *= attn5
			value += attn5 * attn5 * s.extrapolate3(xsb+1, ysb+0, zsb+1, dx5, dy5, dz5)
		}

		// Contribution (0,1,1)
		dx6 := dx0 - 0 - 2*squishConstant3D
		dy6 := dy4
		dz6 := dz5
		attn6 := 2 - dx6*dx6 - dy6*dy6 - dz6*dz6
		if attn6 > 0 {
			attn6 *= attn6
			value += attn6 * attn6 * s.extrapolate3(xsb+0, ysb+1, zsb+1, dx6, dy6, dz6)
		}
	}

	// First extra vertex
	attnExt0 := 2 - dxExt0*dxExt0 - dyExt0*dyExt0 - dzExt0*dzExt0
	if attnExt0 > 0 {
		attnExt0 *= attnExt0
		value += attnExt0 * attnExt0 * s.extrapolate3(xsvExt0, ysvExt0, zsvExt0, dxExt0, dyExt0, dzExt0)
	}

	// Second extra vertex
	attnExt1 := 2 - dxExt1*dxExt1 - dyExt1*dyExt1 - dzExt1*dzExt1
	if attnExt1 > 0 {
		attnExt1 *= attnExt1
		value += attnExt1 * attnExt1 * s.extrapolate3(xsvExt1, ysvExt1, zsvExt1, dxExt1, dyExt1, dzExt1)
	}

	return value / normConstant3D
}

// Eval4 returns a random noise value in four dimensions.
func (s *noise) Eval4(x, y, z, w float64) float64 {
	// Place input coordinates on simplectic honeycomb.
	stretchOffset := (x + y + z + w) * stretchConstant4D
	xs := x + stretchOffset
	ys := y + stretchOffset
	zs := z + stretchOffset
	ws := w + stretchOffset

	// Floor to get simplectic honeycomb coordinates of rhombo-hypercube super-cell origin.
	xsb := int32(math.Floor(xs))
	ysb := int32(math.Floor(ys))
	zsb := int32(math.Floor(zs))
	wsb := int32(math.Floor(ws))

	// Skew out to get actual coordinates of stretched rhombo-hypercube origin. We'll need these later.
	squishOffset := float64(xsb+ysb+zsb+wsb) * squishConstant4D
	xb := float64(xsb) + squishOffset
	yb := float64(ysb) + squishOffset
	zb := float64(zsb) + squishOffset
	wb := float64(wsb) + squishOffset

	// Compute simplectic honeycomb coordinates relative to rhombo-hypercube origin.
	xins := xs - float64(xsb)
	yins := ys - float64(ysb)
	zins := zs - float64(zsb)
	wins := ws - float64(wsb)

	// Sum those together to get a value that determines which region we're in.
	inSum := xins + yins + zins + wins

	// Positions relative to origin point.
	dx0 := x - xb
	dy0 := y - yb
	dz0 := z - zb
	dw0 := w - wb

	// We'll be defining these inside the next block and using them afterwards.
	var dxExt0, dyExt0, dzExt0, dwExt0 float64
	var dxExt1, dyExt1, dzExt1, dwExt1 float64
	var dxExt2, dyExt2, dzExt2, dwExt2 float64
	var xsvExt0, ysvExt0, zsvExt0, wsvExt0 int32
	var xsvExt1, ysvExt1, zsvExt1, wsvExt1 int32
	var xsvExt2, ysvExt2, zsvExt2, wsvExt2 int32

	var value float64 = 0
	if inSum <= 1 { // We're inside the pentachoron (4-Simplex) at (0,0,0,0)
		// Determine which two of (0,0,0,1), (0,0,1,0), (0,1,0,0), (1,0,0,0) are closest.
		var aPoint byte = 0x01
		aScore := xins
		var bPoint byte = 0x02
		bScore := yins
		if aScore >= bScore && zins > bScore {
			bScore = zins
			bPoint = 0x04
		} else if aScore < bScore && zins > aScore {
			aScore = zins
			aPoint = 0x04
		}
		if aScore >= bScore && wins > bScore {
			bScore = wins
			bPoint = 0x08
		} else if aScore < bScore && wins > aScore {
			aScore = wins
			aPoint = 0x08
		}

		// Now we determine the three lattice points not part of the pentachoron that may contribute.
		// This depends on the closest two pentachoron vertices, including (0,0,0,0)
		uins := 1 - inSum
		if uins > aScore || uins > bScore { // (0,0,0,0) is one of the closest two pentachoron vertices.
			var c byte
			// Our other closest vertex is the closest out of a and b.
			if bScore > aScore {
				c = bPoint
			} else {
				c = aPoint
			}
			if (c & 0x01) == 0 {
				xsvExt0 = xsb - 1
				xsvExt2 = xsb
				xsvExt1 = xsvExt2
				dxExt0 = dx0 + 1
				dxExt2 = dx0
				dxExt1 = dxExt2
			} else {
				xsvExt2 = xsb + 1
				xsvExt1 = xsvExt2
				xsvExt0 = xsvExt1
				dxExt2 = dx0 - 1
				dxExt1 = dxExt2
				dxExt0 = dxExt1
			}

			if (c & 0x02) == 0 {
				ysvExt2 = ysb
				ysvExt1 = ysvExt2
				ysvExt0 = ysvExt1
				dyExt2 = dy0
				dyExt1 = dyExt2
				dyExt0 = dyExt1
				if (c & 0x01) == 0x01 {
					ysvExt0 -= 1
					dyExt0 += 1
				} else {
					ysvExt1 -= 1
					dyExt1 += 1
				}
			} else {
				ysvExt2 = ysb + 1
				ysvExt1 = ysvExt2
				ysvExt0 = ysvExt1
				dyExt2 = dy0 - 1
				dyExt1 = dyExt2
				dyExt0 = dyExt1
			}

			if (c & 0x04) == 0 {
				zsvExt2 = zsb
				zsvExt1 = zsvExt2
				zsvExt0 = zsvExt1
				dzExt2 = dz0
				dzExt1 = dzExt2
				dzExt0 = dzExt1
				if (c & 0x03) != 0 {
					if (c & 0x03) == 0x03 {
						zsvExt0 -= 1
						dzExt0 += 1
					} else {
						zsvExt1 -= 1
						dzExt1 += 1
					}
				} else {
					zsvExt2 -= 1
					dzExt2 += 1
				}
			} else {
				zsvExt2 = zsb + 1
				zsvExt1 = zsvExt2
				zsvExt0 = zsvExt1
				dzExt2 = dz0 - 1
				dzExt1 = dzExt2
				dzExt0 = dzExt1
			}

			if (c & 0x08) == 0 {
				wsvExt1 = wsb
				wsvExt0 = wsvExt1
				wsvExt2 = wsb - 1
				dwExt1 = dw0
				dwExt0 = dwExt1
				dwExt2 = dw0 + 1
			} else {
				wsvExt2 = wsb + 1
				wsvExt1 = wsvExt2
				wsvExt0 = wsvExt1
				dwExt2 = dw0 - 1
				dwExt1 = dwExt2
				dwExt0 = dwExt1
			}
		} else { // (0,0,0,0) is not one of the closest two pentachoron vertices.
			c := aPoint | bPoint // Our three extra vertices are determined by the closest two.

			if (c & 0x01) == 0 {
				xsvExt2 = xsb
				xsvExt0 = xsvExt2
				xsvExt1 = xsb - 1
				dxExt0 = dx0 - 2*squishConstant4D
				dxExt1 = dx0 + 1 - squishConstant4D
				dxExt2 = dx0 - squishConstant4D
			} else {
				xsvExt2 = xsb + 1
				xsvExt1 = xsvExt2
				xsvExt0 = xsvExt1
				dxExt0 = dx0 - 1 - 2*squishConstant4D
				dxExt2 = dx0 - 1 - squishConstant4D
				dxExt1 = dxExt2
			}

			if (c & 0x02) == 0 {
				ysvExt2 = ysb
				ysvExt1 = ysvExt2
				ysvExt0 = ysvExt1
				dyExt0 = dy0 - 2*squishConstant4D
				dyExt2 = dy0 - squishConstant4D
				dyExt1 = dyExt2
				if (c & 0x01) == 0x01 {
					ysvExt1 -= 1
					dyExt1 += 1
				} else {
					ysvExt2 -= 1
					dyExt2 += 1
				}
			} else {
				ysvExt2 = ysb + 1
				ysvExt1 = ysvExt2
				ysvExt0 = ysvExt1
				dyExt0 = dy0 - 1 - 2*squishConstant4D
				dyExt2 = dy0 - 1 - squishConstant4D
				dyExt1 = dyExt2
			}

			if (c & 0x04) == 0 {
				zsvExt2 = zsb
				zsvExt1 = zsvExt2
				zsvExt0 = zsvExt1
				dzExt0 = dz0 - 2*squishConstant4D
				dzExt2 = dz0 - squishConstant4D
				dzExt1 = dzExt2
				if (c & 0x03) == 0x03 {
					zsvExt1 -= 1
					dzExt1 += 1
				} else {
					zsvExt2 -= 1
					dzExt2 += 1
				}
			} else {
				zsvExt2 = zsb + 1
				zsvExt1 = zsvExt2
				zsvExt0 = zsvExt1
				dzExt0 = dz0 - 1 - 2*squishConstant4D
				dzExt2 = dz0 - 1 - squishConstant4D
				dzExt1 = dzExt2
			}

			if (c & 0x08) == 0 {
				wsvExt1 = wsb
				wsvExt0 = wsvExt1
				wsvExt2 = wsb - 1
				dwExt0 = dw0 - 2*squishConstant4D
				dwExt1 = dw0 - squishConstant4D
				dwExt2 = dw0 + 1 - squishConstant4D
			} else {
				wsvExt2 = wsb + 1
				wsvExt1 = wsvExt2
				wsvExt0 = wsvExt1
				dwExt0 = dw0 - 1 - 2*squishConstant4D
				dwExt2 = dw0 - 1 - squishConstant4D
				dwExt1 = dwExt2
			}
		}

		// Contribution (0,0,0,0)
		attn0 := 2 - dx0*dx0 - dy0*dy0 - dz0*dz0 - dw0*dw0
		if attn0 > 0 {
			attn0 *= attn0
			value += attn0 * attn0 * s.extrapolate4(xsb+0, ysb+0, zsb+0, wsb+0, dx0, dy0, dz0, dw0)
		}

		// Contribution (1,0,0,0)
		dx1 := dx0 - 1 - squishConstant4D
		dy1 := dy0 - 0 - squishConstant4D
		dz1 := dz0 - 0 - squishConstant4D
		dw1 := dw0 - 0 - squishConstant4D
		attn1 := 2 - dx1*dx1 - dy1*dy1 - dz1*dz1 - dw1*dw1
		if attn1 > 0 {
			attn1 *= attn1
			value += attn1 * attn1 * s.extrapolate4(xsb+1, ysb+0, zsb+0, wsb+0, dx1, dy1, dz1, dw1)
		}

		// Contribution (0,1,0,0)
		dx2 := dx0 - 0 - squishConstant4D
		dy2 := dy0 - 1 - squishConstant4D
		dz2 := dz1
		dw2 := dw1
		attn2 := 2 - dx2*dx2 - dy2*dy2 - dz2*dz2 - dw2*dw2
		if attn2 > 0 {
			attn2 *= attn2
			value += attn2 * attn2 * s.extrapolate4(xsb+0, ysb+1, zsb+0, wsb+0, dx2, dy2, dz2, dw2)
		}

		// Contribution (0,0,1,0)
		dx3 := dx2
		dy3 := dy1
		dz3 := dz0 - 1 - squishConstant4D
		dw3 := dw1
		attn3 := 2 - dx3*dx3 - dy3*dy3 - dz3*dz3 - dw3*dw3
		if attn3 > 0 {
			attn3 *= attn3
			value += attn3 * attn3 * s.extrapolate4(xsb+0, ysb+0, zsb+1, wsb+0, dx3, dy3, dz3, dw3)
		}

		// Contribution (0,0,0,1)
		dx4 := dx2
		dy4 := dy1
		dz4 := dz1
		dw4 := dw0 - 1 - squishConstant4D
		attn4 := 2 - dx4*dx4 - dy4*dy4 - dz4*dz4 - dw4*dw4
		if attn4 > 0 {
			attn4 *= attn4
			value += attn4 * attn4 * s.extrapolate4(xsb+0, ysb+0, zsb+0, wsb+1, dx4, dy4, dz4, dw4)
		}
	} else if inSum >= 3 { // We're inside the pentachoron (4-Simplex) at (1,1,1,1)
		// Determine which two of (1,1,1,0), (1,1,0,1), (1,0,1,1), (0,1,1,1) are closest.
		var aPoint byte = 0x0E
		aScore := xins
		var bPoint byte = 0x0D
		bScore := yins
		if aScore <= bScore && zins < bScore {
			bScore = zins
			bPoint = 0x0B
		} else if aScore > bScore && zins < aScore {
			aScore = zins
			aPoint = 0x0B
		}
		if aScore <= bScore && wins < bScore {
			bScore = wins
			bPoint = 0x07
		} else if aScore > bScore && wins < aScore {
			aScore = wins
			aPoint = 0x07
		}

		// Now we determine the three lattice points not part of the pentachoron that may contribute.
		// This depends on the closest two pentachoron vertices, including (0,0,0,0)
		uins := 4 - inSum
		if uins < aScore || uins < bScore { // (1,1,1,1) is one of the closest two pentachoron vertices.
			var c byte
			// Our other closest vertex is the closest out of a and b.
			if bScore < aScore {
				c = bPoint
			} else {
				c = aPoint
			}

			if (c & 0x01) != 0 {
				xsvExt0 = xsb + 2
				xsvExt2 = xsb + 1
				xsvExt1 = xsvExt2
				dxExt0 = dx0 - 2 - 4*squishConstant4D
				dxExt2 = dx0 - 1 - 4*squishConstant4D
				dxExt1 = dxExt2
			} else {
				xsvExt2 = xsb
				xsvExt1 = xsvExt2
				xsvExt0 = xsvExt1
				dxExt2 = dx0 - 4*squishConstant4D
				dxExt1 = dxExt2
				dxExt0 = dxExt1
			}

			if (c & 0x02) != 0 {
				ysvExt2 = ysb + 1
				ysvExt1 = ysvExt2
				ysvExt0 = ysvExt1
				dyExt2 = dy0 - 1 - 4*squishConstant4D
				dyExt1 = dyExt2
				dyExt0 = dyExt1
				if (c & 0x01) != 0 {
					ysvExt1 += 1
					dyExt1 -= 1
				} else {
					ysvExt0 += 1
					dyExt0 -= 1
				}
			} else {
				ysvExt2 = ysb
				ysvExt1 = ysvExt2
				ysvExt0 = ysvExt1
				dyExt2 = dy0 - 4*squishConstant4D
				dyExt1 = dyExt2
				dyExt0 = dyExt1
			}

			if (c & 0x04) != 0 {
				zsvExt2 = zsb + 1
				zsvExt1 = zsvExt2
				zsvExt0 = zsvExt1
				dzExt2 = dz0 - 1 - 4*squishConstant4D
				dzExt1 = dzExt2
				dzExt0 = dzExt1
				if (c & 0x03) != 0x03 {
					if (c & 0x03) == 0 {
						zsvExt0 += 1
						dzExt0 -= 1
					} else {
						zsvExt1 += 1
						dzExt1 -= 1
					}
				} else {
					zsvExt2 += 1
					dzExt2 -= 1
				}
			} else {
				zsvExt2 = zsb
				zsvExt1 = zsvExt2
				zsvExt0 = zsvExt1
				dzExt2 = dz0 - 4*squishConstant4D
				dzExt1 = dzExt2
				dzExt0 = dzExt1
			}

			if (c & 0x08) != 0 {
				wsvExt1 = wsb + 1
				wsvExt0 = wsvExt1
				wsvExt2 = wsb + 2
				dwExt1 = dw0 - 1 - 4*squishConstant4D
				dwExt0 = dwExt1
				dwExt2 = dw0 - 2 - 4*squishConstant4D
			} else {
				wsvExt2 = wsb
				wsvExt1 = wsvExt2
				wsvExt0 = wsvExt1
				dwExt2 = dw0 - 4*squishConstant4D
				dwExt1 = dwExt2
				dwExt0 = dwExt1
			}
		} else { // (1,1,1,1) is not one of the closest two pentachoron vertices.
			c := aPoint & bPoint // Our three extra vertices are determined by the closest two.

			if (c & 0x01) != 0 {
				xsvExt2 = xsb + 1
				xsvExt0 = xsvExt2
				xsvExt1 = xsb + 2
				dxExt0 = dx0 - 1 - 2*squishConstant4D
				dxExt1 = dx0 - 2 - 3*squishConstant4D
				dxExt2 = dx0 - 1 - 3*squishConstant4D
			} else {
				xsvExt2 = xsb
				xsvExt1 = xsvExt2
				xsvExt0 = xsvExt1
				dxExt0 = dx0 - 2*squishConstant4D
				dxExt2 = dx0 - 3*squishConstant4D
				dxExt1 = dxExt2
			}

			if (c & 0x02) != 0 {
				ysvExt2 = ysb + 1
				ysvExt1 = ysvExt2
				ysvExt0 = ysvExt1
				dyExt0 = dy0 - 1 - 2*squishConstant4D
				dyExt2 = dy0 - 1 - 3*squishConstant4D
				dyExt1 = dyExt2
				if (c & 0x01) != 0 {
					ysvExt2 += 1
					dyExt2 -= 1
				} else {
					ysvExt1 += 1
					dyExt1 -= 1
				}
			} else {
				ysvExt2 = ysb
				ysvExt1 = ysvExt2
				ysvExt0 = ysvExt1
				dyExt0 = dy0 - 2*squishConstant4D
				dyExt2 = dy0 - 3*squishConstant4D
				dyExt1 = dyExt2
			}

			if (c & 0x04) != 0 {
				zsvExt2 = zsb + 1
				zsvExt1 = zsvExt2
				zsvExt0 = zsvExt1
				dzExt0 = dz0 - 1 - 2*squishConstant4D
				dzExt2 = dz0 - 1 - 3*squishConstant4D
				dzExt1 = dzExt2
				if (c & 0x03) != 0 {
					zsvExt2 += 1
					dzExt2 -= 1
				} else {
					zsvExt1 += 1
					dzExt1 -= 1
				}
			} else {
				zsvExt2 = zsb
				zsvExt1 = zsvExt2
				zsvExt0 = zsvExt1
				dzExt0 = dz0 - 2*squishConstant4D
				dzExt2 = dz0 - 3*squishConstant4D
				dzExt1 = dzExt2
			}

			if (c & 0x08) != 0 {
				wsvExt1 = wsb + 1
				wsvExt0 = wsvExt1
				wsvExt2 = wsb + 2
				dwExt0 = dw0 - 1 - 2*squishConstant4D
				dwExt1 = dw0 - 1 - 3*squishConstant4D
				dwExt2 = dw0 - 2 - 3*squishConstant4D
			} else {
				wsvExt2 = wsb
				wsvExt1 = wsvExt2
				wsvExt0 = wsvExt1
				dwExt0 = dw0 - 2*squishConstant4D
				dwExt2 = dw0 - 3*squishConstant4D
				dwExt1 = dwExt2
			}
		}

		// Contribution (1,1,1,0)
		dx4 := dx0 - 1 - 3*squishConstant4D
		dy4 := dy0 - 1 - 3*squishConstant4D
		dz4 := dz0 - 1 - 3*squishConstant4D
		dw4 := dw0 - 3*squishConstant4D
		attn4 := 2 - dx4*dx4 - dy4*dy4 - dz4*dz4 - dw4*dw4
		if attn4 > 0 {
			attn4 *= attn4
			value += attn4 * attn4 * s.extrapolate4(xsb+1, ysb+1, zsb+1, wsb+0, dx4, dy4, dz4, dw4)
		}

		// Contribution (1,1,0,1)
		dx3 := dx4
		dy3 := dy4
		dz3 := dz0 - 3*squishConstant4D
		dw3 := dw0 - 1 - 3*squishConstant4D
		attn3 := 2 - dx3*dx3 - dy3*dy3 - dz3*dz3 - dw3*dw3
		if attn3 > 0 {
			attn3 *= attn3
			value += attn3 * attn3 * s.extrapolate4(xsb+1, ysb+1, zsb+0, wsb+1, dx3, dy3, dz3, dw3)
		}

		// Contribution (1,0,1,1)
		dx2 := dx4
		dy2 := dy0 - 3*squishConstant4D
		dz2 := dz4
		dw2 := dw3
		attn2 := 2 - dx2*dx2 - dy2*dy2 - dz2*dz2 - dw2*dw2
		if attn2 > 0 {
			attn2 *= attn2
			value += attn2 * attn2 * s.extrapolate4(xsb+1, ysb+0, zsb+1, wsb+1, dx2, dy2, dz2, dw2)
		}

		// Contribution (0,1,1,1)
		dx1 := dx0 - 3*squishConstant4D
		dz1 := dz4
		dy1 := dy4
		dw1 := dw3
		attn1 := 2 - dx1*dx1 - dy1*dy1 - dz1*dz1 - dw1*dw1
		if attn1 > 0 {
			attn1 *= attn1
			value += attn1 * attn1 * s.extrapolate4(xsb+0, ysb+1, zsb+1, wsb+1, dx1, dy1, dz1, dw1)
		}

		// Contribution (1,1,1,1)
		dx0 = dx0 - 1 - 4*squishConstant4D
		dy0 = dy0 - 1 - 4*squishConstant4D
		dz0 = dz0 - 1 - 4*squishConstant4D
		dw0 = dw0 - 1 - 4*squishConstant4D
		attn0 := 2 - dx0*dx0 - dy0*dy0 - dz0*dz0 - dw0*dw0
		if attn0 > 0 {
			attn0 *= attn0
			value += attn0 * attn0 * s.extrapolate4(xsb+1, ysb+1, zsb+1, wsb+1, dx0, dy0, dz0, dw0)
		}
	} else if inSum <= 2 { // We're inside the first dispentachoron (Rectified 4-Simplex)
		var aScore, bScore float64
		var aPoint, bPoint byte

		aIsBiggerSide := true
		bIsBiggerSide := true

		// Decide between (1,1,0,0) and (0,0,1,1)
		if xins+yins > zins+wins {
			aScore = xins + yins
			aPoint = 0x03
		} else {
			aScore = zins + wins
			aPoint = 0x0C
		}

		// Decide between (1,0,1,0) and (0,1,0,1)
		if xins+zins > yins+wins {
			bScore = xins + zins
			bPoint = 0x05
		} else {
			bScore = yins + wins
			bPoint = 0x0A
		}

		// Closer between (1,0,0,1) and (0,1,1,0) will replace the further of a and b, if closer.
		if xins+wins > yins+zins {
			score := xins + wins
			if aScore >= bScore && score > bScore {
				bScore = score
				bPoint = 0x09
			} else if aScore < bScore && score > aScore {
				aScore = score
				aPoint = 0x09
			}
		} else {
			score := yins + zins
			if aScore >= bScore && score > bScore {
				bScore = score
				bPoint = 0x06
			} else if aScore < bScore && score > aScore {
				aScore = score
				aPoint = 0x06
			}
		}

		// Decide if (1,0,0,0) is closer.
		p1 := 2 - inSum + xins
		if aScore >= bScore && p1 > bScore {
			bScore = p1
			bPoint = 0x01
			bIsBiggerSide = false
		} else if aScore < bScore && p1 > aScore {
			aScore = p1
			aPoint = 0x01
			aIsBiggerSide = false
		}

		// Decide if (0,1,0,0) is closer.
		p2 := 2 - inSum + yins
		if aScore >= bScore && p2 > bScore {
			bScore = p2
			bPoint = 0x02
			bIsBiggerSide = false
		} else if aScore < bScore && p2 > aScore {
			aScore = p2
			aPoint = 0x02
			aIsBiggerSide = false
		}

		// Decide if (0,0,1,0) is closer.
		p3 := 2 - inSum + zins
		if aScore >= bScore && p3 > bScore {
			bScore = p3
			bPoint = 0x04
			bIsBiggerSide = false
		} else if aScore < bScore && p3 > aScore {
			aScore = p3
			aPoint = 0x04
			aIsBiggerSide = false
		}

		// Decide if (0,0,0,1) is closer.
		p4 := 2 - inSum + wins
		if aScore >= bScore && p4 > bScore {
			bPoint = 0x08
			bIsBiggerSide = false
		} else if aScore < bScore && p4 > aScore {
			aPoint = 0x08
			aIsBiggerSide = false
		}

		// Where each of the two closest points are determines how the extra three vertices are calculated.
		if aIsBiggerSide == bIsBiggerSide {
			if aIsBiggerSide { // Both closest points on the bigger side
				c1 := aPoint | bPoint
				c2 := aPoint & bPoint
				if (c1 & 0x01) == 0 {
					xsvExt0 = xsb
					xsvExt1 = xsb - 1
					dxExt0 = dx0 - 3*squishConstant4D
					dxExt1 = dx0 + 1 - 2*squishConstant4D
				} else {
					xsvExt1 = xsb + 1
					xsvExt0 = xsvExt1
					dxExt0 = dx0 - 1 - 3*squishConstant4D
					dxExt1 = dx0 - 1 - 2*squishConstant4D
				}

				if (c1 & 0x02) == 0 {
					ysvExt0 = ysb
					ysvExt1 = ysb - 1
					dyExt0 = dy0 - 3*squishConstant4D
					dyExt1 = dy0 + 1 - 2*squishConstant4D
				} else {
					ysvExt1 = ysb + 1
					ysvExt0 = ysvExt1
					dyExt0 = dy0 - 1 - 3*squishConstant4D
					dyExt1 = dy0 - 1 - 2*squishConstant4D
				}

				if (c1 & 0x04) == 0 {
					zsvExt0 = zsb
					zsvExt1 = zsb - 1
					dzExt0 = dz0 - 3*squishConstant4D
					dzExt1 = dz0 + 1 - 2*squishConstant4D
				} else {
					zsvExt1 = zsb + 1
					zsvExt0 = zsvExt1
					dzExt0 = dz0 - 1 - 3*squishConstant4D
					dzExt1 = dz0 - 1 - 2*squishConstant4D
				}

				if (c1 & 0x08) == 0 {
					wsvExt0 = wsb
					wsvExt1 = wsb - 1
					dwExt0 = dw0 - 3*squishConstant4D
					dwExt1 = dw0 + 1 - 2*squishConstant4D
				} else {
					wsvExt1 = wsb + 1
					wsvExt0 = wsvExt1
					dwExt0 = dw0 - 1 - 3*squishConstant4D
					dwExt1 = dw0 - 1 - 2*squishConstant4D
				}

				// One combination is a permutation of (0,0,0,2) based on c2
				xsvExt2 = xsb
				ysvExt2 = ysb
				zsvExt2 = zsb
				wsvExt2 = wsb
				dxExt2 = dx0 - 2*squishConstant4D
				dyExt2 = dy0 - 2*squishConstant4D
				dzExt2 = dz0 - 2*squishConstant4D
				dwExt2 = dw0 - 2*squishConstant4D
				if (c2 & 0x01) != 0 {
					xsvExt2 += 2
					dxExt2 -= 2
				} else if (c2 & 0x02) != 0 {
					ysvExt2 += 2
					dyExt2 -= 2
				} else if (c2 & 0x04) != 0 {
					zsvExt2 += 2
					dzExt2 -= 2
				} else {
					wsvExt2 += 2
					dwExt2 -= 2
				}

			} else { // Both closest points on the smaller side
				// One of the two extra points is (0,0,0,0)
				xsvExt2 = xsb
				ysvExt2 = ysb
				zsvExt2 = zsb
				wsvExt2 = wsb
				dxExt2 = dx0
				dyExt2 = dy0
				dzExt2 = dz0
				dwExt2 = dw0

				// Other two points are based on the omitted axes.
				c := aPoint | bPoint

				if (c & 0x01) == 0 {
					xsvExt0 = xsb - 1
					xsvExt1 = xsb
					dxExt0 = dx0 + 1 - squishConstant4D
					dxExt1 = dx0 - squishConstant4D
				} else {
					xsvExt1 = xsb + 1
					xsvExt0 = xsvExt1
					dxExt1 = dx0 - 1 - squishConstant4D
					dxExt0 = dxExt1
				}

				if (c & 0x02) == 0 {
					ysvExt1 = ysb
					ysvExt0 = ysvExt1
					dyExt1 = dy0 - squishConstant4D
					dyExt0 = dyExt1
					if (c & 0x01) == 0x01 {
						ysvExt0 -= 1
						dyExt0 += 1
					} else {
						ysvExt1 -= 1
						dyExt1 += 1
					}
				} else {
					ysvExt1 = ysb + 1
					ysvExt0 = ysvExt1
					dyExt1 = dy0 - 1 - squishConstant4D
					dyExt0 = dyExt1
				}

				if (c & 0x04) == 0 {
					zsvExt1 = zsb
					zsvExt0 = zsvExt1
					dzExt1 = dz0 - squishConstant4D
					dzExt0 = dzExt1
					if (c & 0x03) == 0x03 {
						zsvExt0 -= 1
						dzExt0 += 1
					} else {
						zsvExt1 -= 1
						dzExt1 += 1
					}
				} else {
					zsvExt1 = zsb + 1
					zsvExt0 = zsvExt1
					dzExt1 = dz0 - 1 - squishConstant4D
					dzExt0 = dzExt1
				}

				if (c & 0x08) == 0 {
					wsvExt0 = wsb
					wsvExt1 = wsb - 1
					dwExt0 = dw0 - squishConstant4D
					dwExt1 = dw0 + 1 - squishConstant4D
				} else {
					wsvExt1 = wsb + 1
					wsvExt0 = wsvExt1
					dwExt1 = dw0 - 1 - squishConstant4D
					dwExt0 = dwExt1
				}

			}
		} else { // One point on each "side"
			var c1, c2 byte
			if aIsBiggerSide {
				c1 = aPoint
				c2 = bPoint
			} else {
				c1 = bPoint
				c2 = aPoint
			}

			// Two contributions are the bigger-sided point with each 0 replaced with -1.
			if (c1 & 0x01) == 0 {
				xsvExt0 = xsb - 1
				xsvExt1 = xsb
				dxExt0 = dx0 + 1 - squishConstant4D
				dxExt1 = dx0 - squishConstant4D
			} else {
				xsvExt1 = xsb + 1
				xsvExt0 = xsvExt1
				dxExt1 = dx0 - 1 - squishConstant4D
				dxExt0 = dxExt1
			}

			if (c1 & 0x02) == 0 {
				ysvExt1 = ysb
				ysvExt0 = ysvExt1
				dyExt1 = dy0 - squishConstant4D
				dyExt0 = dyExt1
				if (c1 & 0x01) == 0x01 {
					ysvExt0 -= 1
					dyExt0 += 1
				} else {
					ysvExt1 -= 1
					dyExt1 += 1
				}
			} else {
				ysvExt1 = ysb + 1
				ysvExt0 = ysvExt1
				dyExt1 = dy0 - 1 - squishConstant4D
				dyExt0 = dyExt1
			}

			if (c1 & 0x04) == 0 {
				zsvExt1 = zsb
				zsvExt0 = zsvExt1
				dzExt1 = dz0 - squishConstant4D
				dzExt0 = dzExt1
				if (c1 & 0x03) == 0x03 {
					zsvExt0 -= 1
					dzExt0 += 1
				} else {
					zsvExt1 -= 1
					dzExt1 += 1
				}
			} else {
				zsvExt1 = zsb + 1
				zsvExt0 = zsvExt1
				dzExt1 = dz0 - 1 - squishConstant4D
				dzExt0 = dzExt1
			}

			if (c1 & 0x08) == 0 {
				wsvExt0 = wsb
				wsvExt1 = wsb - 1
				dwExt0 = dw0 - squishConstant4D
				dwExt1 = dw0 + 1 - squishConstant4D
			} else {
				wsvExt1 = wsb + 1
				wsvExt0 = wsvExt1
				dwExt1 = dw0 - 1 - squishConstant4D
				dwExt0 = dwExt1
			}

			// One contribution is a permutation of (0,0,0,2) based on the smaller-sided point
			xsvExt2 = xsb
			ysvExt2 = ysb
			zsvExt2 = zsb
			wsvExt2 = wsb
			dxExt2 = dx0 - 2*squishConstant4D
			dyExt2 = dy0 - 2*squishConstant4D
			dzExt2 = dz0 - 2*squishConstant4D
			dwExt2 = dw0 - 2*squishConstant4D
			if (c2 & 0x01) != 0 {
				xsvExt2 += 2
				dxExt2 -= 2
			} else if (c2 & 0x02) != 0 {
				ysvExt2 += 2
				dyExt2 -= 2
			} else if (c2 & 0x04) != 0 {
				zsvExt2 += 2
				dzExt2 -= 2
			} else {
				wsvExt2 += 2
				dwExt2 -= 2
			}
		}

		// Contribution (1,0,0,0)
		dx1 := dx0 - 1 - squishConstant4D
		dy1 := dy0 - 0 - squishConstant4D
		dz1 := dz0 - 0 - squishConstant4D
		dw1 := dw0 - 0 - squishConstant4D
		attn1 := 2 - dx1*dx1 - dy1*dy1 - dz1*dz1 - dw1*dw1
		if attn1 > 0 {
			attn1 *= attn1
			value += attn1 * attn1 * s.extrapolate4(xsb+1, ysb+0, zsb+0, wsb+0, dx1, dy1, dz1, dw1)
		}

		// Contribution (0,1,0,0)
		dx2 := dx0 - 0 - squishConstant4D
		dy2 := dy0 - 1 - squishConstant4D
		dz2 := dz1
		dw2 := dw1
		attn2 := 2 - dx2*dx2 - dy2*dy2 - dz2*dz2 - dw2*dw2
		if attn2 > 0 {
			attn2 *= attn2
			value += attn2 * attn2 * s.extrapolate4(xsb+0, ysb+1, zsb+0, wsb+0, dx2, dy2, dz2, dw2)
		}

		// Contribution (0,0,1,0)
		dx3 := dx2
		dy3 := dy1
		dz3 := dz0 - 1 - squishConstant4D
		dw3 := dw1
		attn3 := 2 - dx3*dx3 - dy3*dy3 - dz3*dz3 - dw3*dw3
		if attn3 > 0 {
			attn3 *= attn3
			value += attn3 * attn3 * s.extrapolate4(xsb+0, ysb+0, zsb+1, wsb+0, dx3, dy3, dz3, dw3)
		}

		// Contribution (0,0,0,1)
		dx4 := dx2
		dy4 := dy1
		dz4 := dz1
		dw4 := dw0 - 1 - squishConstant4D
		attn4 := 2 - dx4*dx4 - dy4*dy4 - dz4*dz4 - dw4*dw4
		if attn4 > 0 {
			attn4 *= attn4
			value += attn4 * attn4 * s.extrapolate4(xsb+0, ysb+0, zsb+0, wsb+1, dx4, dy4, dz4, dw4)
		}

		// Contribution (1,1,0,0)
		dx5 := dx0 - 1 - 2*squishConstant4D
		dy5 := dy0 - 1 - 2*squishConstant4D
		dz5 := dz0 - 0 - 2*squishConstant4D
		dw5 := dw0 - 0 - 2*squishConstant4D
		attn5 := 2 - dx5*dx5 - dy5*dy5 - dz5*dz5 - dw5*dw5
		if attn5 > 0 {
			attn5 *= attn5
			value += attn5 * attn5 * s.extrapolate4(xsb+1, ysb+1, zsb+0, wsb+0, dx5, dy5, dz5, dw5)
		}

		// Contribution (1,0,1,0)
		dx6 := dx0 - 1 - 2*squishConstant4D
		dy6 := dy0 - 0 - 2*squishConstant4D
		dz6 := dz0 - 1 - 2*squishConstant4D
		dw6 := dw0 - 0 - 2*squishConstant4D
		attn6 := 2 - dx6*dx6 - dy6*dy6 - dz6*dz6 - dw6*dw6
		if attn6 > 0 {
			attn6 *= attn6
			value += attn6 * attn6 * s.extrapolate4(xsb+1, ysb+0, zsb+1, wsb+0, dx6, dy6, dz6, dw6)
		}

		// Contribution (1,0,0,1)
		dx7 := dx0 - 1 - 2*squishConstant4D
		dy7 := dy0 - 0 - 2*squishConstant4D
		dz7 := dz0 - 0 - 2*squishConstant4D
		dw7 := dw0 - 1 - 2*squishConstant4D
		attn7 := 2 - dx7*dx7 - dy7*dy7 - dz7*dz7 - dw7*dw7
		if attn7 > 0 {
			attn7 *= attn7
			value += attn7 * attn7 * s.extrapolate4(xsb+1, ysb+0, zsb+0, wsb+1, dx7, dy7, dz7, dw7)
		}

		// Contribution (0,1,1,0)
		dx8 := dx0 - 0 - 2*squishConstant4D
		dy8 := dy0 - 1 - 2*squishConstant4D
		dz8 := dz0 - 1 - 2*squishConstant4D
		dw8 := dw0 - 0 - 2*squishConstant4D
		attn8 := 2 - dx8*dx8 - dy8*dy8 - dz8*dz8 - dw8*dw8
		if attn8 > 0 {
			attn8 *= attn8
			value += attn8 * attn8 * s.extrapolate4(xsb+0, ysb+1, zsb+1, wsb+0, dx8, dy8, dz8, dw8)
		}

		// Contribution (0,1,0,1)
		dx9 := dx0 - 0 - 2*squishConstant4D
		dy9 := dy0 - 1 - 2*squishConstant4D
		dz9 := dz0 - 0 - 2*squishConstant4D
		dw9 := dw0 - 1 - 2*squishConstant4D
		attn9 := 2 - dx9*dx9 - dy9*dy9 - dz9*dz9 - dw9*dw9
		if attn9 > 0 {
			attn9 *= attn9
			value += attn9 * attn9 * s.extrapolate4(xsb+0, ysb+1, zsb+0, wsb+1, dx9, dy9, dz9, dw9)
		}

		// Contribution (0,0,1,1)
		dx10 := dx0 - 0 - 2*squishConstant4D
		dy10 := dy0 - 0 - 2*squishConstant4D
		dz10 := dz0 - 1 - 2*squishConstant4D
		dw10 := dw0 - 1 - 2*squishConstant4D
		attn10 := 2 - dx10*dx10 - dy10*dy10 - dz10*dz10 - dw10*dw10
		if attn10 > 0 {
			attn10 *= attn10
			value += attn10 * attn10 * s.extrapolate4(xsb+0, ysb+0, zsb+1, wsb+1, dx10, dy10, dz10, dw10)
		}
	} else { // We're inside the second dispentachoron (Rectified 4-Simplex)
		var aScore, bScore float64
		var aPoint, bPoint byte

		aIsBiggerSide := true
		bIsBiggerSide := true

		// Decide between (0,0,1,1) and (1,1,0,0)
		if xins+yins < zins+wins {
			aScore = xins + yins
			aPoint = 0x0C
		} else {
			aScore = zins + wins
			aPoint = 0x03
		}

		// Decide between (0,1,0,1) and (1,0,1,0)
		if xins+zins < yins+wins {
			bScore = xins + zins
			bPoint = 0x0A
		} else {
			bScore = yins + wins
			bPoint = 0x05
		}

		// Closer between (0,1,1,0) and (1,0,0,1) will replace the further of a and b, if closer.
		if xins+wins < yins+zins {
			score := xins + wins
			if aScore <= bScore && score < bScore {
				bScore = score
				bPoint = 0x06
			} else if aScore > bScore && score < aScore {
				aScore = score
				aPoint = 0x06
			}
		} else {
			score := yins + zins
			if aScore <= bScore && score < bScore {
				bScore = score
				bPoint = 0x09
			} else if aScore > bScore && score < aScore {
				aScore = score
				aPoint = 0x09
			}
		}

		// Decide if (0,1,1,1) is closer.
		p1 := 3 - inSum + xins
		if aScore <= bScore && p1 < bScore {
			bScore = p1
			bPoint = 0x0E
			bIsBiggerSide = false
		} else if aScore > bScore && p1 < aScore {
			aScore = p1
			aPoint = 0x0E
			aIsBiggerSide = false
		}

		// Decide if (1,0,1,1) is closer.
		p2 := 3 - inSum + yins
		if aScore <= bScore && p2 < bScore {
			bScore = p2
			bPoint = 0x0D
			bIsBiggerSide = false
		} else if aScore > bScore && p2 < aScore {
			aScore = p2
			aPoint = 0x0D
			aIsBiggerSide = false
		}

		// Decide if (1,1,0,1) is closer.
		p3 := 3 - inSum + zins
		if aScore <= bScore && p3 < bScore {
			bScore = p3
			bPoint = 0x0B
			bIsBiggerSide = false
		} else if aScore > bScore && p3 < aScore {
			aScore = p3
			aPoint = 0x0B
			aIsBiggerSide = false
		}

		// Decide if (1,1,1,0) is closer.
		p4 := 3 - inSum + wins
		if aScore <= bScore && p4 < bScore {
			bPoint = 0x07
			bIsBiggerSide = false
		} else if aScore > bScore && p4 < aScore {
			aPoint = 0x07
			aIsBiggerSide = false
		}

		// Where each of the two closest points are determines how the extra three vertices are calculated.
		if aIsBiggerSide == bIsBiggerSide {
			if aIsBiggerSide { // Both closest points on the bigger side
				c1 := aPoint & bPoint
				c2 := aPoint | bPoint

				// Two contributions are permutations of (0,0,0,1) and (0,0,0,2) based on c1
				xsvExt1 = xsb
				xsvExt0 = xsvExt1
				ysvExt1 = ysb
				ysvExt0 = ysvExt1
				zsvExt1 = zsb
				zsvExt0 = zsvExt1
				wsvExt1 = wsb
				wsvExt0 = wsvExt1
				dxExt0 = dx0 - squishConstant4D
				dyExt0 = dy0 - squishConstant4D
				dzExt0 = dz0 - squishConstant4D
				dwExt0 = dw0 - squishConstant4D
				dxExt1 = dx0 - 2*squishConstant4D
				dyExt1 = dy0 - 2*squishConstant4D
				dzExt1 = dz0 - 2*squishConstant4D
				dwExt1 = dw0 - 2*squishConstant4D
				if (c1 & 0x01) != 0 {
					xsvExt0 += 1
					dxExt0 -= 1
					xsvExt1 += 2
					dxExt1 -= 2
				} else if (c1 & 0x02) != 0 {
					ysvExt0 += 1
					dyExt0 -= 1
					ysvExt1 += 2
					dyExt1 -= 2
				} else if (c1 & 0x04) != 0 {
					zsvExt0 += 1
					dzExt0 -= 1
					zsvExt1 += 2
					dzExt1 -= 2
				} else {
					wsvExt0 += 1
					dwExt0 -= 1
					wsvExt1 += 2
					dwExt1 -= 2
				}

				// One contribution is a permutation of (1,1,1,-1) based on c2
				xsvExt2 = xsb + 1
				ysvExt2 = ysb + 1
				zsvExt2 = zsb + 1
				wsvExt2 = wsb + 1
				dxExt2 = dx0 - 1 - 2*squishConstant4D
				dyExt2 = dy0 - 1 - 2*squishConstant4D
				dzExt2 = dz0 - 1 - 2*squishConstant4D
				dwExt2 = dw0 - 1 - 2*squishConstant4D
				if (c2 & 0x01) == 0 {
					xsvExt2 -= 2
					dxExt2 += 2
				} else if (c2 & 0x02) == 0 {
					ysvExt2 -= 2
					dyExt2 += 2
				} else if (c2 & 0x04) == 0 {
					zsvExt2 -= 2
					dzExt2 += 2
				} else {
					wsvExt2 -= 2
					dwExt2 += 2
				}
			} else { // Both closest points on the smaller side
				// One of the two extra points is (1,1,1,1)
				xsvExt2 = xsb + 1
				ysvExt2 = ysb + 1
				zsvExt2 = zsb + 1
				wsvExt2 = wsb + 1
				dxExt2 = dx0 - 1 - 4*squishConstant4D
				dyExt2 = dy0 - 1 - 4*squishConstant4D
				dzExt2 = dz0 - 1 - 4*squishConstant4D
				dwExt2 = dw0 - 1 - 4*squishConstant4D

				// Other two points are based on the shared axes.
				c := aPoint & bPoint

				if (c & 0x01) != 0 {
					xsvExt0 = xsb + 2
					xsvExt1 = xsb + 1
					dxExt0 = dx0 - 2 - 3*squishConstant4D
					dxExt1 = dx0 - 1 - 3*squishConstant4D
				} else {
					xsvExt1 = xsb
					xsvExt0 = xsvExt1
					dxExt1 = dx0 - 3*squishConstant4D
					dxExt0 = dxExt1
				}

				if (c & 0x02) != 0 {
					ysvExt1 = ysb + 1
					ysvExt0 = ysvExt1
					dyExt1 = dy0 - 1 - 3*squishConstant4D
					dyExt0 = dyExt1
					if (c & 0x01) == 0 {
						ysvExt0 += 1
						dyExt0 -= 1
					} else {
						ysvExt1 += 1
						dyExt1 -= 1
					}
				} else {
					ysvExt1 = ysb
					ysvExt0 = ysvExt1
					dyExt1 = dy0 - 3*squishConstant4D
					dyExt0 = dyExt1
				}

				if (c & 0x04) != 0 {
					zsvExt1 = zsb + 1
					zsvExt0 = zsvExt1
					dzExt1 = dz0 - 1 - 3*squishConstant4D
					dzExt0 = dzExt1
					if (c & 0x03) == 0 {
						zsvExt0 += 1
						dzExt0 -= 1
					} else {
						zsvExt1 += 1
						dzExt1 -= 1
					}
				} else {
					zsvExt1 = zsb
					zsvExt0 = zsvExt1
					dzExt1 = dz0 - 3*squishConstant4D
					dzExt0 = dzExt1
				}

				if (c & 0x08) != 0 {
					wsvExt0 = wsb + 1
					wsvExt1 = wsb + 2
					dwExt0 = dw0 - 1 - 3*squishConstant4D
					dwExt1 = dw0 - 2 - 3*squishConstant4D
				} else {
					wsvExt1 = wsb
					wsvExt0 = wsvExt1
					dwExt1 = dw0 - 3*squishConstant4D
					dwExt0 = dwExt1
				}
			}
		} else { // One point on each "side"
			var c1, c2 byte
			if aIsBiggerSide {
				c1 = aPoint
				c2 = bPoint
			} else {
				c1 = bPoint
				c2 = aPoint
			}

			// Two contributions are the bigger-sided point with each 1 replaced with 2.
			if (c1 & 0x01) != 0 {
				xsvExt0 = xsb + 2
				xsvExt1 = xsb + 1
				dxExt0 = dx0 - 2 - 3*squishConstant4D
				dxExt1 = dx0 - 1 - 3*squishConstant4D
			} else {
				xsvExt1 = xsb
				xsvExt0 = xsvExt1
				dxExt1 = dx0 - 3*squishConstant4D
				dxExt0 = dxExt1
			}

			if (c1 & 0x02) != 0 {
				ysvExt1 = ysb + 1
				ysvExt0 = ysvExt1
				dyExt1 = dy0 - 1 - 3*squishConstant4D
				dyExt0 = dyExt1
				if (c1 & 0x01) == 0 {
					ysvExt0 += 1
					dyExt0 -= 1
				} else {
					ysvExt1 += 1
					dyExt1 -= 1
				}
			} else {
				ysvExt1 = ysb
				ysvExt0 = ysvExt1
				dyExt1 = dy0 - 3*squishConstant4D
				dyExt0 = dyExt1
			}

			if (c1 & 0x04) != 0 {
				zsvExt1 = zsb + 1
				zsvExt0 = zsvExt1
				dzExt1 = dz0 - 1 - 3*squishConstant4D
				dzExt0 = dzExt1
				if (c1 & 0x03) == 0 {
					zsvExt0 += 1
					dzExt0 -= 1
				} else {
					zsvExt1 += 1
					dzExt1 -= 1
				}
			} else {
				zsvExt1 = zsb
				zsvExt0 = zsvExt1
				dzExt1 = dz0 - 3*squishConstant4D
				dzExt0 = dzExt1
			}

			if (c1 & 0x08) != 0 {
				wsvExt0 = wsb + 1
				wsvExt1 = wsb + 2
				dwExt0 = dw0 - 1 - 3*squishConstant4D
				dwExt1 = dw0 - 2 - 3*squishConstant4D
			} else {
				wsvExt1 = wsb
				wsvExt0 = wsvExt1
				dwExt1 = dw0 - 3*squishConstant4D
				dwExt0 = dwExt1
			}

			// One contribution is a permutation of (1,1,1,-1) based on the smaller-sided point
			xsvExt2 = xsb + 1
			ysvExt2 = ysb + 1
			zsvExt2 = zsb + 1
			wsvExt2 = wsb + 1
			dxExt2 = dx0 - 1 - 2*squishConstant4D
			dyExt2 = dy0 - 1 - 2*squishConstant4D
			dzExt2 = dz0 - 1 - 2*squishConstant4D
			dwExt2 = dw0 - 1 - 2*squishConstant4D
			if (c2 & 0x01) == 0 {
				xsvExt2 -= 2
				dxExt2 += 2
			} else if (c2 & 0x02) == 0 {
				ysvExt2 -= 2
				dyExt2 += 2
			} else if (c2 & 0x04) == 0 {
				zsvExt2 -= 2
				dzExt2 += 2
			} else {
				wsvExt2 -= 2
				dwExt2 += 2
			}
		}

		// Contribution (1,1,1,0)
		dx4 := dx0 - 1 - 3*squishConstant4D
		dy4 := dy0 - 1 - 3*squishConstant4D
		dz4 := dz0 - 1 - 3*squishConstant4D
		dw4 := dw0 - 3*squishConstant4D
		attn4 := 2 - dx4*dx4 - dy4*dy4 - dz4*dz4 - dw4*dw4
		if attn4 > 0 {
			attn4 *= attn4
			value += attn4 * attn4 * s.extrapolate4(xsb+1, ysb+1, zsb+1, wsb+0, dx4, dy4, dz4, dw4)
		}

		// Contribution (1,1,0,1)
		dx3 := dx4
		dy3 := dy4
		dz3 := dz0 - 3*squishConstant4D
		dw3 := dw0 - 1 - 3*squishConstant4D
		attn3 := 2 - dx3*dx3 - dy3*dy3 - dz3*dz3 - dw3*dw3
		if attn3 > 0 {
			attn3 *= attn3
			value += attn3 * attn3 * s.extrapolate4(xsb+1, ysb+1, zsb+0, wsb+1, dx3, dy3, dz3, dw3)
		}

		// Contribution (1,0,1,1)
		dx2 := dx4
		dy2 := dy0 - 3*squishConstant4D
		dz2 := dz4
		dw2 := dw3
		attn2 := 2 - dx2*dx2 - dy2*dy2 - dz2*dz2 - dw2*dw2
		if attn2 > 0 {
			attn2 *= attn2
			value += attn2 * attn2 * s.extrapolate4(xsb+1, ysb+0, zsb+1, wsb+1, dx2, dy2, dz2, dw2)
		}

		// Contribution (0,1,1,1)
		dx1 := dx0 - 3*squishConstant4D
		dz1 := dz4
		dy1 := dy4
		dw1 := dw3
		attn1 := 2 - dx1*dx1 - dy1*dy1 - dz1*dz1 - dw1*dw1
		if attn1 > 0 {
			attn1 *= attn1
			value += attn1 * attn1 * s.extrapolate4(xsb+0, ysb+1, zsb+1, wsb+1, dx1, dy1, dz1, dw1)
		}

		// Contribution (1,1,0,0)
		dx5 := dx0 - 1 - 2*squishConstant4D
		dy5 := dy0 - 1 - 2*squishConstant4D
		dz5 := dz0 - 0 - 2*squishConstant4D
		dw5 := dw0 - 0 - 2*squishConstant4D
		attn5 := 2 - dx5*dx5 - dy5*dy5 - dz5*dz5 - dw5*dw5
		if attn5 > 0 {
			attn5 *= attn5
			value += attn5 * attn5 * s.extrapolate4(xsb+1, ysb+1, zsb+0, wsb+0, dx5, dy5, dz5, dw5)
		}

		// Contribution (1,0,1,0)
		dx6 := dx0 - 1 - 2*squishConstant4D
		dy6 := dy0 - 0 - 2*squishConstant4D
		dz6 := dz0 - 1 - 2*squishConstant4D
		dw6 := dw0 - 0 - 2*squishConstant4D
		attn6 := 2 - dx6*dx6 - dy6*dy6 - dz6*dz6 - dw6*dw6
		if attn6 > 0 {
			attn6 *= attn6
			value += attn6 * attn6 * s.extrapolate4(xsb+1, ysb+0, zsb+1, wsb+0, dx6, dy6, dz6, dw6)
		}

		// Contribution (1,0,0,1)
		dx7 := dx0 - 1 - 2*squishConstant4D
		dy7 := dy0 - 0 - 2*squishConstant4D
		dz7 := dz0 - 0 - 2*squishConstant4D
		dw7 := dw0 - 1 - 2*squishConstant4D
		attn7 := 2 - dx7*dx7 - dy7*dy7 - dz7*dz7 - dw7*dw7
		if attn7 > 0 {
			attn7 *= attn7
			value += attn7 * attn7 * s.extrapolate4(xsb+1, ysb+0, zsb+0, wsb+1, dx7, dy7, dz7, dw7)
		}

		// Contribution (0,1,1,0)
		dx8 := dx0 - 0 - 2*squishConstant4D
		dy8 := dy0 - 1 - 2*squishConstant4D
		dz8 := dz0 - 1 - 2*squishConstant4D
		dw8 := dw0 - 0 - 2*squishConstant4D
		attn8 := 2 - dx8*dx8 - dy8*dy8 - dz8*dz8 - dw8*dw8
		if attn8 > 0 {
			attn8 *= attn8
			value += attn8 * attn8 * s.extrapolate4(xsb+0, ysb+1, zsb+1, wsb+0, dx8, dy8, dz8, dw8)
		}

		// Contribution (0,1,0,1)
		dx9 := dx0 - 0 - 2*squishConstant4D
		dy9 := dy0 - 1 - 2*squishConstant4D
		dz9 := dz0 - 0 - 2*squishConstant4D
		dw9 := dw0 - 1 - 2*squishConstant4D
		attn9 := 2 - dx9*dx9 - dy9*dy9 - dz9*dz9 - dw9*dw9
		if attn9 > 0 {
			attn9 *= attn9
			value += attn9 * attn9 * s.extrapolate4(xsb+0, ysb+1, zsb+0, wsb+1, dx9, dy9, dz9, dw9)
		}

		// Contribution (0,0,1,1)
		dx10 := dx0 - 0 - 2*squishConstant4D
		dy10 := dy0 - 0 - 2*squishConstant4D
		dz10 := dz0 - 1 - 2*squishConstant4D
		dw10 := dw0 - 1 - 2*squishConstant4D
		attn10 := 2 - dx10*dx10 - dy10*dy10 - dz10*dz10 - dw10*dw10
		if attn10 > 0 {
			attn10 *= attn10
			value += attn10 * attn10 * s.extrapolate4(xsb+0, ysb+0, zsb+1, wsb+1, dx10, dy10, dz10, dw10)
		}
	}

	// First extra vertex
	attnExt0 := 2 - dxExt0*dxExt0 - dyExt0*dyExt0 - dzExt0*dzExt0 - dwExt0*dwExt0
	if attnExt0 > 0 {
		attnExt0 *= attnExt0
		value += attnExt0 * attnExt0 * s.extrapolate4(xsvExt0, ysvExt0, zsvExt0, wsvExt0, dxExt0, dyExt0, dzExt0, dwExt0)
	}

	// Second extra vertex
	attnExt1 := 2 - dxExt1*dxExt1 - dyExt1*dyExt1 - dzExt1*dzExt1 - dwExt1*dwExt1
	if attnExt1 > 0 {
		attnExt1 *= attnExt1
		value += attnExt1 * attnExt1 * s.extrapolate4(xsvExt1, ysvExt1, zsvExt1, wsvExt1, dxExt1, dyExt1, dzExt1, dwExt1)
	}

	// Third extra vertex
	attnExt2 := 2 - dxExt2*dxExt2 - dyExt2*dyExt2 - dzExt2*dzExt2 - dwExt2*dwExt2
	if attnExt2 > 0 {
		attnExt2 *= attnExt2
		value += attnExt2 * attnExt2 * s.extrapolate4(xsvExt2, ysvExt2, zsvExt2, wsvExt2, dxExt2, dyExt2, dzExt2, dwExt2)
	}

	return value / normConstant4D
}
