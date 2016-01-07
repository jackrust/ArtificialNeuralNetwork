using System;
using System.Collections.Generic;
using System.Linq;

namespace ArtificialNeuralNetwork
{
    public class Data
    {
        public List<DataPoint> DataPoints;
        public Func<List<double>, List<double>, bool, bool> SuccessCondition;

        public Data()
        {
            DataPoints = new List<DataPoint>();
        }
        public List<List<double>> Inputs()
        {
            return DataPoints.Select(dp => dp.Inputs).ToList();
        }
        public List<List<double>> Outputs()
        {
            return DataPoints.Select(dp => dp.Outputs).ToList();
        }
    }

    public class DataPoint
    {
        public string Reference;
        public List<double> Inputs;
        public List<double> Outputs;

        public DataPoint()
        {
            Inputs = new List<double>();
            Outputs = new List<double>();
        }
        public DataPoint(List<double> input, List<double> output)
        {
            Inputs = input;
            Outputs = output;
        }
    }
}
