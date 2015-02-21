using System;
using System.Collections.Generic;
using System.Linq;
using Utilities;

namespace ArtificialNeuralNetwork
{
    public class FeedForwardNetwork : Network
    {
        public FeedForwardNetwork(int inputsNo, IReadOnlyCollection<int> hiddenNos, int outputsNo)
        {
            CreateInputs(inputsNo);
            CreateHiddenLayers(hiddenNos.Count);
            var layer = 1;
            foreach (var h in hiddenNos)
            {
                CreateHiddenLayer(layer, h);
                layer++;
            }
            CreateOutputs(outputsNo);
        }

        protected override List<double> Run()
        {
            //If inputs and/or outputs aren't set up then bail now
            if (!InputsValid() || !OutputsValid())
                return null;

            //Pull the output from the output neurons

            return ONeurons.Select(o => o.GetOutput()).ToList();
        }

        protected override void Backpropagate(Dynamic oNeuron, double err)
        {
            oNeuron.Backpropagate(err);
        }

        public override void TrainHoldBestInvestigate(List<List<double>> inputs, List<List<double>> targets)
        {
            Epochs = 0;
            var minima = 0;
            double minError = -1;
            double maxError = -1;
            double prevError = -1;
            var log = new List<List<double>>();

            var bestWeights = GetWeights();
            do
            {
                Error = TrainEpoch(inputs, targets) / inputs.Count;
                Epochs++;
                minima++;

                if (Error < minError || minError < 0)
                {
                    minima = 0;
                    minError = Error;
                    bestWeights = GetWeights();
                }

                if (Error > maxError)
                {
                    maxError = Error;
                }

                if (Error > prevError)
                {
                    AdjustLearningRateDown();
                }
                prevError = Error;
                log.Add(new List<double>() { Epochs, minima, Error, minError, maxError });
            } while (Error > TargetError && minima < MaxMinima && Epochs < MaxEpochs);
            SetWeights(bestWeights);
            RecordLog(log);
            var rankings = RankInputs();
            RecordRankings(rankings);
        }

        public override List<double> GetWeights()
        {
            var weights = new List<double>();
            foreach (var h in HLayers.SelectMany(l => l))
            {
                weights.AddRange(h.GetWeights());
            }
            foreach (var o in ONeurons)
            {
                weights.AddRange(o.GetWeights());
            }
            return weights;
        }

        public override void SetWeights(List<double> weights)
        {
            var index = 0;
            foreach (var l in HLayers)
            {
                foreach (var h in l)
                {
                    h.SetWeights(weights.GetRange(index, h.Dendrites.Count()));
                    index += h.Dendrites.Count();
                }
            }
            foreach (var o in ONeurons)
            {
                o.SetWeights(weights.GetRange(index, o.Dendrites.Count()));
                index += o.Dendrites.Count();
            }
        }
    }
}
