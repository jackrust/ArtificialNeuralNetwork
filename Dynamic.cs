using System;
using System.Collections.Generic;
using System.Linq;
using Utilities;

namespace ArtificialNeuralNetwork
{
    public class Dynamic:Neuron
    {
        public static double MinLearningRate = 1.0 / Math.Pow(2.0, 8.0);
        public static double MaxLearningRate = 0.25;
        //TODO:Add momentum
        public double LearningRate = MaxLearningRate;

	    public Dynamic() {}
	    public Dynamic(List<Neuron> inputs):base(inputs) {}
	    public Dynamic(String name):base(name) {}

        public Dynamic(List<Dendrite> dendrites, String name, double threshold)
            : base(dendrites, name, threshold) {}


	    public override double GetOutput() {
		    var sum = Dendrites.Aggregate<Dendrite, double>(0, (current, d) => current + (d.GetSignal()));
	        return Function(sum);
	    }

	    /**
	     * backpropagate
	     * sends weighted error back through the network
	     * re-weights input based on error.
	     * @param error (double)
	     */
	    public override void Backpropagate(double error) {
		    //http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
	        foreach (var d in Dendrites)
	        {
	            d.Neuron.Backpropagate(error*d.Weight);
	        }
            foreach (var d in Dendrites)
            {
	            Reweight(d, error);
            } 
	    }
	
	    /**
	     * reweight
	     * Sets the weight of the given dendrite based on its error.
	     * @param dendrite (Dendrite) to be reweighted
	     * @param error (double)
	     */
	    private void Reweight(Dendrite dendrite, double error) {
            var weight = dendrite.Weight + LearningRate * error * Derivative(GetOutput()) * dendrite.Neuron.GetOutput();
		    dendrite.Weight = weight;
		
	    }
	
	    /**
	     * function
	     * computes output using exponential
	     * @return double result
	     */
	    protected double Function(double x) {
		    var result = 1/(1 + Math.Exp(-x));
            return result;
	    }
	
	    /**
	     * function
	     * derivative of function
	     * @return double result
	     */
	    protected double Derivative(double x) {
		    var result =  Math.Exp(-x)/((1 + Math.Exp(-x))*(1 + Math.Exp(-x)));
            return result;
	    }

        public void HalveLearningRate()
        {
            if (Math.Round(LearningRate, 6) > Math.Round(MinLearningRate, 6))
                LearningRate /= 2;
        }

        //TOTO: merge with below?
        public void PlugIn(List<Dynamic> neurons)
        {
            foreach (var n in neurons)
            {
                var temp = n;
                foreach (var d in Dendrites.Where(d => d.Neuron.Name == temp.Name))
                {
                    d.Neuron = n;
                }
            }
        }

        public void PlugIn(List<Input> neurons)
        {
            foreach (var n in neurons)
            {
                var temp = n;
                foreach (var d in Dendrites.Where(d => d.Neuron.Name == temp.Name))
                {
                    d.Neuron = n;
                }
            }
        }

        public string Stringify()
        {
            var s = "";
            s += "<threshold>" + Threshold + "</threshold>";
            s += "<name>" + Name + "</name>";

            s += "<dendrites>";
            foreach (var d in Dendrites)
            {
                s += "<dendrite>";
                s += d.Stringify();
                s += "</dendrite>";
            }
            s += "</dendrite>";
            return s;
        }

        public static Dynamic Objectify(string str)
        {
            var threshold = Convert.ToDouble(Stringy.SplitOn(str, "threshold")[0]);
            var name = Stringy.SplitOn(str, "name")[0];

            var ds = Stringy.SplitOn(Stringy.SplitOn(str, "dendrites")[0], "dendrite");
            var dendrites = new List<Dendrite>();
            for (var d = 0; d < ds.Count(); d++ )
            {
                dendrites[d] = Dendrite.Objectify(ds[d]);
            }
            return new Dynamic(dendrites, name, threshold);
        }

        internal static List<List<Dynamic>> Copy(List<List<Dynamic>> input)
        {
            return input.Select(Copy).ToList();
        }

        public static List<Dynamic> Copy(List<Dynamic> input)
        {
            return input.Select(Copy).ToList();
        }

        public static Dynamic Copy(Dynamic input)
        {
            return new Dynamic(Dendrite.Copy(input.Dendrites), input.Name, input.Threshold);
        }
    }
}
