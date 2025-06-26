-- File: src/LSTM.hs
module LSTM where

import Math
import Control.Monad (replicateM)
import System.Random (randomRIO)

data LSTMState = LSTMState {
    hidden :: Vector,
    cell   :: Vector
} deriving Show

data LSTM = LSTM {
    wf :: Matrix, bf :: Vector,
    wi :: Matrix, bi :: Vector,
    wc :: Matrix, bc :: Vector,
    wo :: Matrix, bo :: Vector
} deriving Show

type Gradients = (Matrix, Vector, Matrix, Vector, Matrix, Vector, Matrix, Vector)

-- | Gradient clipping
clipGrads :: Double -> Gradients -> Gradients
clipGrads thresh (dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo) =
    (clipM dWf, clipV dbf, clipM dWi, clipV dbi, clipM dWc, clipV dbc, clipM dWo, clipV dbo)
  where
    clipM = map (map (clip thresh))
    clipV = map (clip thresh)
    clip t x = max (-t) (min t x)

-- | Initialize LSTM
initLSTM :: Int -> Int -> IO LSTM
initLSTM inputSize hiddenSize = do
  let zSize = inputSize + hiddenSize
  wf' <- randMatrix hiddenSize zSize
  bf' <- randVector hiddenSize
  wi' <- randMatrix hiddenSize zSize
  bi' <- randVector hiddenSize
  wc' <- randMatrix hiddenSize zSize
  bc' <- randVector hiddenSize
  wo' <- randMatrix hiddenSize zSize
  bo' <- randVector hiddenSize
  return $ LSTM wf' bf' wi' bi' wc' bc' wo' bo'

-- | Forward pass (with cache)
type Cache = (Vector, Vector, Vector, Vector, Vector, Vector, Vector, Vector, Vector)

lstmForwardWithCache :: LSTM -> Vector -> LSTMState -> (LSTMState, Cache)
lstmForwardWithCache lstm x (LSTMState hPrev cPrev) =
  let z = concatVec hPrev x
      f = sigmoidVec $ matVec (wf lstm) z `addVec` (bf lstm)
      i = sigmoidVec $ matVec (wi lstm) z `addVec` (bi lstm)
      cTilde = tanhVec $ matVec (wc lstm) z `addVec` (bc lstm)
      c = zipWith3 (\ft ct itc -> ft * ct + itc) f cPrev (zipWith (*) i cTilde)
      o = sigmoidVec $ matVec (wo lstm) z `addVec` (bo lstm)
      h = zipWith (*) o (tanhVec c)
  in (LSTMState h c, (z, f, i, cTilde, c, o, h, cPrev, hPrev))

-- | Backward step
lstmBackwardStep
  :: LSTM -> Vector -> Vector -> Cache -> (Gradients, Vector, Vector)
lstmBackwardStep lstm dh dc (z, f, i, cTilde, c, o, h, cPrev, hPrev) =
  let tanhC = tanhVec c
      do_ = mulVec dh tanhC
      dc_total = addVec dc (mulVec dh (mulVec o (tanhDerivVec c)))
      df = mulVec dc_total cPrev
      di = mulVec dc_total cTilde
      dcTilde = mulVec dc_total i

      df_raw = mulVec df (mulVec f (map (1 -) f))
      di_raw = mulVec di (mulVec i (map (1 -) i))
      do_raw = mulVec do_ (mulVec o (map (1 -) o))
      dcTilde_raw = mulVec dcTilde (map (\x -> 1 - x ** 2) cTilde)

      dWf = outer df_raw z
      dWi = outer di_raw z
      dWc = outer dcTilde_raw z
      dWo = outer do_raw z

      dbf = df_raw
      dbi = di_raw
      dbc = dcTilde_raw
      dbo = do_raw

      dzf = matVecT (wf lstm) df_raw
      dzi = matVecT (wi lstm) di_raw
      dzc = matVecT (wc lstm) dcTilde_raw
      dzo = matVecT (wo lstm) do_raw

      dz = foldl1 addVec [dzf, dzi, dzc, dzo]
      (dhPrev, _) = splitAt (length hPrev) dz
      dcPrev = mulVec dc_total f
  in ((dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo), dcPrev, dhPrev)

zeroGrads :: Int -> Int -> Gradients
zeroGrads r c = (zeroM r c, zeroV r, zeroM r c, zeroV r, zeroM r c, zeroV r, zeroM r c, zeroV r)

addGrads :: Gradients -> Gradients -> Gradients
addGrads (a1,b1,c1,d1,e1,f1,g1,h1) (a2,b2,c2,d2,e2,f2,g2,h2) =
  ( addMat a1 a2, addVec b1 b2
  , addMat c1 c2, addVec d1 d2
  , addMat e1 e2, addVec f1 f2
  , addMat g1 g2, addVec h1 h2 )

-- | Full BPTT
lstmBackward
  :: LSTM
  -> [Cache]
  -> [Vector]
  -> Gradients
lstmBackward lstm caches dhs =
  let (_, grad, _, _) = foldr step (Nothing, zeroGrads hSize zSize, zeroV hSize, zeroV hSize) (zip dhs caches)
  in grad
  where
    hSize = length (snd3 (head caches))
    zSize = length (fst3 (head caches))
    snd3 (_, b, _, _, _, _, _, _, _) = b
    fst3 (a, _, _, _, _, _, _, _, _) = a

    step (dh, cache) (Nothing, acc, dc, _) =
      let (grad, dc', dh') = lstmBackwardStep lstm dh dc cache
      in (Just True, addGrads acc grad, dc', dh')

    step (dh, cache) (Just _, acc, dc, _) =
      let (grad, dc', dh') = lstmBackwardStep lstm dh dc cache
      in (Just True, addGrads acc grad, dc', dh')
