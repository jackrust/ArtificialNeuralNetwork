using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace ArtificialNeuralNetwork
{
    public class Optimizer
    {
        public const int UpperLimitLayers = 1;
        public const int LowerLimitLayers = 1;
        public const int UpperLimitNeuronsInLayer = 6;
        public const int LowerLimitNeuronsInLayer = 6;
        public const int UpperLimitEpochs = 2000;
        public const int LowerLimitEpochs = 2000;
        public const double UpperLimitTargetError = 0.005;
        public const double LowerLimitTargetError = 0.005;
        public const double TargetErrorStep = 0.005;
        public List<Network.TrainingAlgorithm> Algorithms;

        public string Optimize(Data trainingData, Data testingData, Func<List<double>, List<double>, bool> successCondition, Func<List<double>, List<double>> deconvert)
        {
            Algorithms = new List<Network.TrainingAlgorithm>() { Network.TrainingAlgorithm.HoldBestInvestigate };// Enum.GetValues(typeof(Network.TrainingAlgorithm)).Cast<Network.TrainingAlgorithm>())
            var grapher = new StringBuilder();
            grapher.AppendLine("");
            grapher.AppendLine("Graph data:");
            grapher.AppendLine("Layers|INeurons|Success|Time");

            for (var numLayers = LowerLimitLayers; numLayers < UpperLimitLayers + 1; numLayers++)
            {
                for (var perLayer = LowerLimitNeuronsInLayer; perLayer < (numLayers > 0 ? UpperLimitNeuronsInLayer + 1 : 2); perLayer++)
                {
                    for (var epoch = LowerLimitEpochs; epoch < UpperLimitEpochs + 1; epoch++)
                    {
                        for (var err = LowerLimitTargetError; err < UpperLimitTargetError + 1; err++)
                        {
                            foreach (var algorithm in Algorithms )
                            {
                                grapher.AppendLine(RunTestNetwork(trainingData, testingData, successCondition, deconvert, numLayers, perLayer, algorithm, false));
                                Console.WriteLine(grapher.ToString());
                            }
                        }
                    }
                }
            }
            return grapher.ToString();
        }

        public static string RunTestNetwork(Data trainingData, Data testingData, Func<List<double>, List<double>, bool> successCondition, Func<List<double>, List<double>> deconvert, int numLayers, int perLayer, Network.TrainingAlgorithm algorithm, bool saveReport = true, bool feedforward = true)
        {
            //Create hidden layers
            var hidden = new List<int>();

            for (var i = 0; i < numLayers; i++)
            {
                hidden.Add(perLayer);
            }

            //Create Network
            Network network;
            if (feedforward)
            {
                network = new FeedForwardNetwork(trainingData.Inputs[0].Count, hidden, trainingData.Outputs[0].Count);
            }
            else
            {
                network = new InterconnectedNetwork(trainingData.Inputs[0].Count, perLayer, trainingData.Outputs[0].Count);
            }

            //Start a stopwatch
            var stopWatch = new Stopwatch();
            stopWatch.Start();

            //Train the network
            network.Train(trainingData.Inputs, trainingData.Outputs, algorithm);

            //Stop the stopwatch
            stopWatch.Stop();

            //Test

            if (saveReport)
            {
                SaveReport(testingData, successCondition, deconvert, network);
            }
            var successes = testingData.Inputs.Select(t => network.Run(t)).Where((result, i) => successCondition(result, testingData.Outputs[i])).Count();

            return String.Format("{0}, {1}|{2}|{3}", numLayers, perLayer,
               Math.Round((successes / (double)testingData.Inputs.Count) * 100, 2),
               (double)stopWatch.ElapsedMilliseconds / 1000);
        }

        private static void SaveReport(Data testingData, Func<List<double>, List<double>, bool> successCondition, Func<List<double>, List<double>> deconvert, Network network)
        {
            var report = "";
            for (var i = 0; i < testingData.Inputs.Count; i++)
            {
                var output = network.Run(testingData.Inputs[i]);
                var sccss = successCondition(output, testingData.Outputs[i]) ? 1 : 0;
                report = String.Format("{0}|i{1}|o{2}|t{3}|s|{4}\n", i,
                    testingData.Inputs[i].Aggregate(report, (current, inpt) => current + ("|" + inpt)),
                    deconvert(output).Aggregate(report, (current, otpt) => current + ("|" + otpt)),
                    deconvert(testingData.Outputs[i]).Aggregate(report, (current, trgt) => current + ("|" + trgt)),
                    sccss);
            }

            using (var file = new System.IO.StreamWriter("report.txt"))
            {
                file.WriteLine(report);
            }
        }
    }
}
