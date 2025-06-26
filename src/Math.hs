module Math where

import Data.List (transpose)
import System.Random (randomRIO)

type Vector = [Double]
type Matrix = [Vector]

-- Basic vector ops
addVec, subVec, mulVec :: Vector -> Vector -> Vector
addVec = zipWith (+)
subVec = zipWith (-)
mulVec = zipWith (*)

scalarVec :: Double -> Vector -> Vector
scalarVec s = map (s *)

dot :: Vector -> Vector -> Double
dot v1 v2 = sum $ zipWith (*) v1 v2

norm :: Vector -> Double
norm v = sqrt $ sum $ map (^2) v

-- Matrix-vector
matVec :: Matrix -> Vector -> Vector
matVec m v = map (dot v) (transpose m)

matVecT :: Matrix -> Vector -> Vector
matVecT m v = map (dot v) m

-- Zeros
zeroV :: Int -> Vector
zeroV n = replicate n 0

zeroM :: Int -> Int -> Matrix
zeroM rows cols = replicate rows (zeroV cols)

-- Matrix ops
addMat :: Matrix -> Matrix -> Matrix
addMat = zipWith addVec

subMat :: Matrix -> Matrix -> Matrix
subMat = zipWith (zipWith (-))

scaleMat :: Double -> Matrix -> Matrix
scaleMat scalar = map (map (scalar *))

scaleVec :: Double -> Vector -> Vector
scaleVec scalar = map (scalar *)

-- Activation
sigmoid :: Double -> Double
sigmoid x = 1 / (1 + exp (-x))

dsigmoid :: Double -> Double
dsigmoid y = y * (1 - y)

tanh' :: Double -> Double
tanh' = tanh

dtanh :: Double -> Double
dtanh y = 1 - y * y

-- Softmax
softmax :: [Double] -> [Double]
softmax xs = let exps = map exp xs
                 total = sum exps
             in map (/ total) exps

-- Cosine similarity
cosineSim :: Vector -> Vector -> Double
cosineSim u v = dot u v / (norm u * norm v + 1e-8)

-- Random init
randVector :: Int -> IO Vector
randVector n = mapM (\_ -> randomRIO (-0.1, 0.1)) [1..n]

randMatrix :: Int -> Int -> IO Matrix
randMatrix r c = mapM (const $ randVector c) [1..r]

concatVec :: Vector -> Vector -> Vector
concatVec = (++)

splitAtVec :: Int -> Vector -> (Vector, Vector)
splitAtVec = splitAt

-- Vectorized activations
sigmoidVec :: Vector -> Vector
sigmoidVec = map sigmoid

tanhVec :: Vector -> Vector
tanhVec = map tanh'

tanhDerivVec :: Vector -> Vector
tanhDerivVec = map dtanh

-- Outer product
outer :: Vector -> Vector -> Matrix
outer u v = [ map (* x) v | x <- u ]

-- Mean squared error between two vectors
mseVec :: Vector -> Vector -> Double
mseVec v1 v2 = sum (zipWith (\a b -> (a - b) ^ 2) v1 v2) / fromIntegral (length v1)

-- Gradient clipping for list of matrices
clipGradients :: Double -> [Matrix] -> [Matrix]
clipGradients threshold mats = map clip mats
  where
    clip m =
      let norm = sqrt . sum . map (^2) $ concat m
      in if norm > threshold
         then map (map (\x -> x * (threshold / norm))) m
         else m
