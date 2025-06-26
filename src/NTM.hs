-- File: src/NTM.hs
module NTM where

import Math
import LSTM
import Memory
import Addressing

import Data.List (zipWith4)

-- | Full NTM State
data NTMState = NTMState {
    controllerState :: LSTMState,
    memoryState     :: MemoryState
} deriving Show

-- | NTM Parameters
data NTM = NTM {
    controller :: LSTM,
    memorySize :: Int,
    slotSize   :: Int
} deriving Show

-- | Forward step for NTM
ntmStep :: NTM -> Vector -> NTMState -> (Vector, NTMState)
ntmStep ntm input (NTMState lstmState memState) =
  let mem      = memory memState
      readVec  = readVector memState
      lstmIn   = concatVec input readVec
      (lstmState', _) = lstmForwardWithCache (controller ntm) lstmIn lstmState
      ctrlOut  = hidden lstmState'

      (key, rest1)    = splitAt (slotSize ntm) ctrlOut
      (gV, rest2)     = splitAt 1 rest1
      (s, rest3)      = splitAt (memorySize ntm) rest2
      (gammaV, rest4) = splitAt 1 rest3
      (erase, add)    = splitAt (slotSize ntm) rest4

      g = head gV
      gamma = head gammaV
      wPrev = writeWeights memState
      w     = addressing key wPrev mem g s gamma

      mem' = writeToMemory mem w erase add
      r    = readFromMemory mem' w

      newMemState = MemoryState mem' w w r
  in (ctrlOut, NTMState lstmState' newMemState)

-- | Forward pass for entire sequence
forwardNTM :: NTM -> [Vector] -> ([Vector], [NTMState])
forwardNTM ntm inputs = go inputs (initialState ntm) []
  where
    go [] _ acc = (reverse (map fst acc), reverse (map snd acc))
    go (x:xs) state acc =
      let (out, state') = ntmStep ntm x state
      in go xs state' ((out, state') : acc)

-- | Backward pass with gradient clipping
backwardNTM :: NTM -> [Vector] -> [Vector] -> [Vector] -> [NTMState] -> NTM
backwardNTM ntm inputs targets outputs states =
  let lstmInputs = zipWith (\x st -> concatVec x (readVector (memoryState st))) inputs states
      lstmCaches = zipWith
                (\x ntmSt -> snd (lstmForwardWithCache (controller ntm) x (controllerState ntmSt)))
                lstmInputs states
      dLoss y t = map (*2) (zipWith (-) y t)
      gradInputs = zipWith dLoss outputs targets
      grads = lstmBackward (controller ntm) lstmCaches gradInputs
      clipped = clipGrads 1.0 grads
  in ntm { controller = applyGrads (controller ntm) clipped 0.01 }

applyGrads :: LSTM -> Gradients -> Double -> LSTM
applyGrads lstm (dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo) lr =
  LSTM
    (addMat (wf lstm) (scaleMat (-lr) dWf))
    (addVec (bf lstm) (scaleVec (-lr) dbf))
    (addMat (wi lstm) (scaleMat (-lr) dWi))
    (addVec (bi lstm) (scaleVec (-lr) dbi))
    (addMat (wc lstm) (scaleMat (-lr) dWc))
    (addVec (bc lstm) (scaleVec (-lr) dbc))
    (addMat (wo lstm) (scaleMat (-lr) dWo))
    (addVec (bo lstm) (scaleVec (-lr) dbo))

mseLoss :: [Vector] -> [Vector] -> Double
mseLoss ys ts = sum (zipWith mseVec ys ts) / fromIntegral (length ys)

-- | Init state
initialState :: NTM -> NTMState
initialState ntm =
  let hSize = 4 * slotSize ntm + 1 + memorySize ntm + 1
      lstmState0 = LSTMState (zeroV hSize) (zeroV hSize)
      memState0 = initMemoryState (memorySize ntm) (slotSize ntm)
  in NTMState lstmState0 memState0

-- | Init NTM with controller
initNTM :: Int -> Int -> Int -> IO (NTM, NTMState)
initNTM inputSize memSize slotSize = do
  let ctrlInputSize = inputSize + slotSize
      ctrlOutputSize = 4 * slotSize + 1 + memSize + 1
  lstm <- initLSTM ctrlInputSize ctrlOutputSize
  let ntm = NTM lstm memSize slotSize
  return (ntm, initialState ntm)
