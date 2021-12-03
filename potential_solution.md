SO
We have G(h(x)) where h(x) is our parameter generator. Since h(x) is a neural net, we also know h'(x). We don't know G'(x) and need to approximate it for backpropagation.

What we'll do is compute `(G(y) - G(y + d)) / d` for several ideas of `d`
Then when we need G'(y) (where y is h(x)), we'll look at the different ideas of `d` and try to find `G'(y + d)` which is closest to the derivative we're required to meet, and then pass the corresponding `d` down the line as the derivative.

The choice of `d` is important, and instead of random choices like [0.312321, 0.14124, 0.8123123] we'll want choices like [0., 0.00123, 0.] since we want _small_ adjustments to the resulting image

Oh, also G(x) must be deterministic