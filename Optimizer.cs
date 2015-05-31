using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace ArtificialNeuralNetwork
{
    public class Optimizer
    {
        public int UpperLimitLayers;
        public int LowerLimitLayers;
        public int UpperLimitNeuronsInLayer;
        public int LowerLimitNeuronsInLayer;
        public int UpperLimitEpochs;
        public int LowerLimitEpochs;
        public double UpperLimitTargetError;
        public double LowerLimitTargetError;
        public double TargetErrorStep;
        public List<TrainingAlgorithmFactory.TrainingAlgorithmType> Algorithms;

        public Optimizer()
        {
            UpperLimitLayers = 1;
            LowerLimitLayers = 1;
            UpperLimitNeuronsInLayer = 1;
            LowerLimitNeuronsInLayer = 1;
            UpperLimitEpochs = 2000;
            LowerLimitEpochs = 2000;
            UpperLimitTargetError = 0.005;
            LowerLimitTargetError = 0.005;
            TargetErrorStep = 0.005;
            Algorithms = new List<TrainingAlgorithmFactory.TrainingAlgorithmType>();
        }

        public string Optimize(Data trainingData, Data testingData, Func<List<double>, List<double>, bool> successCondition, Func<List<double>, List<double>> deconvert)
        {
            Algorithms = new List<TrainingAlgorithmFactory.TrainingAlgorithmType>() { TrainingAlgorithmFactory.TrainingAlgorithmType.HoldBestInvestigate };// Enum.GetValues(typeof(TrainingAlgorithm.TrainingAlgorithmType)).Cast<Network.TrainingAlgorithm>())
            var grapher = new StringBuilder();
            grapher.AppendLine("");
            grapher.AppendLine("Graph data:");
            grapher.AppendLine("id|Layers|INeurons|Success|Time");

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

        public static string RunTestNetwork(Data trainingData, Data testingData, Func<List<double>, List<double>, bool> successCondition, Func<List<double>, List<double>> deconvert, int numLayers, int perLayer, TrainingAlgorithmFactory.TrainingAlgorithmType algorithm, bool saveReport = true)
        {
            //Create hidden layers
            var hidden = new List<int>();

            for (var i = 0; i < numLayers; i++)
            {
                hidden.Add(perLayer);
            }

            //Create Network
            Network network = new Network(trainingData.Inputs[0].Count, hidden, trainingData.Outputs[0].Count);

            //Start a stopwatch
            var stopWatch = new Stopwatch();
            stopWatch.Start();

            //Train the network
            network.Train(trainingData.Inputs, trainingData.Outputs, algorithm);
            Network.Save(network);

            //Stop the stopwatch
            stopWatch.Stop();

            //Test
            if (saveReport)
            {
                SaveReport(testingData, successCondition, deconvert, network);
            }
            var successes = testingData.Inputs.Select(t => network.Run(t)).Where((result, i) => successCondition(result, testingData.Outputs[i])).Count();

            return String.Format("{0}|{1}|{2}|{3}|{4}", network.Id, numLayers, perLayer,
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

            using (var file = new System.IO.StreamWriter("Optimizer/report_" + network.Id + ".txt"))
            {
                file.WriteLine(report);
            }
        }
    }
}
