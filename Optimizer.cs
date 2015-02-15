using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace ArtificialNeuralNetwork
{
    public class Optimizer
    {
        public const int UpperLimitLayers = 2;
        public const int LowerLimitLayers = 1;
        public const int UpperLimitNeuronsInLayer = 5;
        public const int LowerLimitNeuronsInLayer = 1;
        public const int UpperLimitEpochs = 2000;
        public const int LowerLimitEpochs = 2000;
        public const double UpperLimitTargetError = 0.005;
        public const double LowerLimitTargetError = 0.005;
        public const double TargetErrorStep = 0.005;

        public string Optimize(Data trainingData, Data testingData, Func<List<double>, List<double>, bool> successCondition)
        {
            var grapher = new StringBuilder();
            grapher.AppendLine("");
            grapher.AppendLine("Graph data:");
            grapher.AppendLine("Layers|INeurons|Success|Time");

            for (var numLayers = LowerLimitLayers; numLayers < UpperLimitLayers + 1; numLayers++)
            {
                for (var perLayer = LowerLimitNeuronsInLayer; perLayer < UpperLimitNeuronsInLayer + 1; perLayer++)
                {
                    for (var epoch = LowerLimitEpochs; epoch < UpperLimitEpochs + 1; epoch++)
                    {
                        for (var err = LowerLimitTargetError; err < UpperLimitTargetError + 1; err++)
                        {
                            foreach (var algorithm in new List<Network.TrainingAlgorithm>() { Network.TrainingAlgorithm.HoldBestSpiralOut })// Enum.GetValues(typeof(Network.TrainingAlgorithm)).Cast<Network.TrainingAlgorithm>())
                            {
                                grapher.AppendLine(RunTest(trainingData, testingData, successCondition, numLayers, perLayer, algorithm));
                                Console.WriteLine(grapher.ToString());
                            }
                        }
                    }
                }
            }
            return grapher.ToString();
        }

        public static string RunTest(Data trainingData, Data testingData, Func<List<double>, List<double>, bool> successCondition, int numLayers, int perLayer, Network.TrainingAlgorithm algorithm)
        {
            //Create hidden layers
            var hidden = new List<int>();

            for (var i = 0; i < numLayers; i++)
            {
                hidden.Add(perLayer);
            }

            //Create Network
            var network = new Network(trainingData.Inputs[0].Count, hidden, trainingData.Outputs[0].Count);
            //New network with 5 inputs, One hidden layer of 2 neurons, 1 output

            //Start a stopwatch
            var stopWatch = new Stopwatch();
            stopWatch.Start();

            //Train the network
            network.Train(trainingData.Inputs, trainingData.Outputs, algorithm);

            //Stop the stopwatch
            stopWatch.Stop();

            //Test
            var successes = testingData.Inputs.Select(t => network.Run(t)).Where((result, i) => successCondition(result, testingData.Outputs[i])).Count();

            return String.Format("{0}, {1}|{2}|{3}", numLayers, perLayer,
               Math.Round((successes / (double)testingData.Inputs.Count) * 100, 2),
               (double)stopWatch.ElapsedMilliseconds / 1000);
        }

    }
}
