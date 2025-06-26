-- File: src/Addressing.hs
module Addressing where

import Math
import Data.List (zipWith4)
import Data.Function (on)
import Data.Ord (comparing)
import Data.Bifunctor (second)


rotate :: Int -> [a] -> [a]
rotate n xs = take len $ drop (n `mod` len) (cycle xs)
  where len = length xs

-- Interpolation between previous weight and content weights
interpolate :: Double -> Vector -> Vector -> Vector
interpolate g wc wPrev = zipWith (\a b -> g * a + (1 - g) * b) wc wPrev

-- Circular convolution for shifting
circularConvolve :: Vector -> Vector -> Vector
circularConvolve w s = [ sum [ w !! ((i - j) `mod` len) * s !! j | j <- [0..len-1] ] | i <- [0..len-1] ]
  where len = length w

-- Sharpening
sharpen :: Double -> Vector -> Vector
sharpen gamma w = let w' = map (**gamma) w
                      s = sum w'
                  in map (/s) w'

-- Full address weighting
addressing :: Vector     -- ^ Key (k)
           -> Vector     -- ^ Previous weight (w_{t-1})
           -> [Vector]   -- ^ Memory
           -> Double     -- ^ g (interpolation gate)
           -> Vector     -- ^ s (shift weighting)
           -> Double     -- ^ gamma (sharpening)
           -> Vector     -- ^ Final weight w_t
addressing key wPrev mem g s gamma =
  let wc = softmax $ map (cosineSim key) mem
      wg = interpolate g wc wPrev
      wShift = circularConvolve wg s
      wSharp = sharpen gamma wShift
  in wSharp
