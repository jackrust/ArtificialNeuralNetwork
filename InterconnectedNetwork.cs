using System;
using System.Collections.Generic;
using System.Linq;

namespace ArtificialNeuralNetwork
{
    class InterconnectedNetwork : Network
    {
        public List<Neuron> Neurons;

        public InterconnectedNetwork(int inputsNo, int hiddenNo, int outputsNo)
        {
            INeurons = new List<Input>();
            for (var i = 0; i < inputsNo; i++)
                INeurons.Add(new Input("I" + (i + 1)));

            HLayers = new List<List<Dynamic>> {new List<Dynamic>()};
            for (var i = 0; i < hiddenNo; i++)
                HLayers[0].Add(new Dynamic("H" + (i + 1)));

            ONeurons = new List<Dynamic>();
            for (var i = 0; i < outputsNo; i++)
                ONeurons.Add(new Dynamic("O" + (i + 1)));
            Neurons = new List<Neuron>();
            Neurons.AddRange(INeurons);
            Neurons.AddRange(HLayers[0]);
            Neurons.AddRange(ONeurons);

            foreach (var n in Neurons)
            {
                n.AddDendrites(Neurons);
            }
        }

        protected override List<double> Run()
        {
            //If inputs and/or outputs aren't set up then bail now
            if (!InputsValid() || !OutputsValid())
                return null;

            //Pull the output from the output neurons

            return ONeurons.Select(o => o.GetOutput(new NeuralPathway())).ToList();
        }

        protected override void Backpropagate(Dynamic oNeuron, double err)
        {
            oNeuron.Backpropagate(err, new NeuralPathway());
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
        }

        public override List<double> GetWeights()
        {
            var weights = new List<double>();
            foreach (var n in Neurons)
            {
                weights.AddRange(n.GetWeights());
            }
            return weights;
        }

        public override void SetWeights(List<double> weights)
        {
            var index = 0;
            foreach (var n in Neurons)
            {
                n.SetWeights(weights.GetRange(index, n.Dendrites.Count()));
                index += n.Dendrites.Count();
            }
        }
    }
}
