OpenSimplex in Go
=================


[![Go PKG](https://pkg.go.dev/go.sdls.io/opensimplex?status.svg)](https://pkg.go.dev/go.sdls.io/opensimplex)

OpenSimplex noise is a random noise algorithm by Kurt Spencer, made as a
patent-free alternative to Perlin and Simplex noise. This Go port is based on
Kurt's [Java implementation](https://gist.github.com/KdotJPG/b1270127455a94ac5d19).

For an introduction to OpenSimplex noise, see [Kurt Spencer's
post](https://uniblock.tumblr.com/post/97868843242/noise) announcing it. If
you're not familiar with random noise, the Wikipedia post on [Perlin
noise](https://en.wikipedia.org/wiki/Perlin_noise) is a good introduction.


Why not Perlin noise?
---------------------

As Kurt explains [in his
post](https://uniblock.tumblr.com/post/97868843242/noise), Perlin noise tends to
generate noise with noticeable axis-aligned artifacts. Simplex noise fixes
these artifacts, but it's patented. OpenSimplex noise is for people who don't
want to deal with Simplex's patent.

The difference between Perlin and OpenSimplex noise is easiest to see in
pictures. This is Perlin noise, with a noticeable bias towards vertical and
horizontal artifacts:

![Perlin Noise sample](https://67.media.tumblr.com/153d6efb739746f114c52f86391c3932/tumblr_inline_nc4swj5tX51seaucq.png)

Here's what OpenSimplex noise looks like:

![OpenSimplex Noise sample](https://67.media.tumblr.com/6186a25f7bafb258c30101ee3c0c87b4/tumblr_inline_ngubweRMTr1seaucq.png)


Tests
-----------
This implementation of OpenSimplex's tests verify its output against the output
of the reference Java implementation. I haven't run these tests on different
architectures, so results may vary.

License
-------
This code is maintained by SUDOLESS under BSD-4, for the original version of the codebase check
[ojra's repo](https://github.com/ojrac/opensimplex-go)
under the same "license" as Kurt's OpenSimplex - the public domain
"unlicense."
