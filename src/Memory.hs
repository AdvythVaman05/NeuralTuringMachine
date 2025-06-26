-- File: src/Memory.hs
module Memory where

import Math
import Data.List (transpose, maximumBy)
import Data.Ord (comparing)
import Data.Function (on)
import Data.List (zipWith4)

-- | Memory is a list of memory slots (each a vector)
type Memory = [Vector]

data MemoryState = MemoryState {
    memory :: Memory,
    readWeights :: Vector,
    writeWeights :: Vector,
    readVector :: Vector
} deriving Show


-- Content-based addressing
contentWeights :: Vector -> Memory -> Vector
contentWeights key mem = softmax $ map (cosineSim key) mem

-- Read from memory with soft weights
readFromMemory :: Memory -> Vector -> Vector
readFromMemory mem weights = foldl1 addVec $ zipWith scalarVec weights mem

-- Write to memory (erase + add)
writeToMemory :: Memory -> Vector -> Vector -> Vector -> Memory
writeToMemory mem weights erase add =
  zipWith4 (\m w -> \e a ->
      let m' = mulVec (map (1 -) (scalarVec w e)) m
          m'' = addVec m' (scalarVec w a)
      in m'') mem weights (repeat erase) (repeat add)

-- Initialize memory state
initMemoryState :: Int -> Int -> MemoryState
initMemoryState memSize slotSize =
  let mem = replicate memSize (replicate slotSize 0.01)
      w = replicate memSize (1 / fromIntegral memSize)
      r = replicate slotSize 0.0
  in MemoryState mem w w r
