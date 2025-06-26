-- File: src/Train.hs
module Train where

import Math
import LSTM
import Memory
import Addressing
import NTM

import Control.Monad (forM_, when)
import Text.Printf (printf)
import Data.List (intercalate, sort)
import System.IO (appendFile)

-- | Make task: identity copy of a sequence of 0.0 and 1.0
import System.Random (randomRIO)

makeTask :: Int -> IO ([Vector], [Vector])
makeTask len = do
  base <- mapM (\_ -> randomRIO (0.0, 1.0)) [1..len]
  let inputs = map (\x -> [x]) base
      targets = map (\x -> [x]) (sort base)
  return (inputs, targets)


-- | Print vector
printVec :: Vector -> String
printVec v = "[" ++ intercalate ", " (map (printf "%.2f") v) ++ "]"

-- | Train for given number of epochs
trainNTM :: Int -> IO ()
trainNTM epochs = do
  let inputSize = 1
      memorySlots = 20
      vectorSize = 10

  (ntm, _) <- initNTM inputSize memorySlots vectorSize
  (inp, tgt) <- makeTask 10

  let loop 0 ntm' = return ntm'
      loop n ntm' = do
        let (loss, newNTM, outs) = trainStepVerbose ntm' inp tgt
        printf "Epoch %d | MSE: %.6f\n" (epochs - n + 1) loss
        putStrLn $ "Input:     " ++ unwords (map printVec inp)
        putStrLn $ "Target:    " ++ unwords (map printVec tgt)
        putStrLn $ "Predicted: " ++ unwords (map printVec outs)
        putStrLn "----"

        when (n `mod` 10 == 0) $ do
          let csvLine = intercalate "," $
                          concatMap (map (printf "%.4f")) inp ++
                          concatMap (map (printf "%.4f")) tgt ++
                          concatMap (map (printf "%.4f")) outs
          appendFile "training_log.csv" (csvLine ++ "\n")

        loop (n - 1) newNTM

  _ <- loop epochs ntm
  return ()


-- | Train step
trainStepVerbose :: NTM -> [Vector] -> [Vector] -> (Double, NTM, [Vector])
trainStepVerbose ntm input target =
  let (outputs, caches) = forwardNTM ntm input
      loss = mseLoss outputs target
      ntm' = backwardNTM ntm input target outputs caches
  in (loss, ntm', outputs)
