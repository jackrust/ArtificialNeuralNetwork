using System;
using System.Collections.Generic;
using System.Linq;
using Utilities;

namespace ArtificialNeuralNetwork
{
    public class Input:Neuron
    {
        public static double DefaultValue = 1;
        public double Value = DefaultValue;


        public Input() { this.Dendrites = new List<Dendrite>(); }


        public Input(String name)
            : this(name, DefaultValue)
        {
            this.Dendrites = new List<Dendrite>();
	    }

	    public Input(double value):this(DefaultName, value)
        {
            this.Dendrites = new List<Dendrite>();
	    }
	
	    public Input(String name, double value):base(name)
        {
		    Value = value;
            this.Dendrites = new List<Dendrite>();
	    }

        public Input(List<Dendrite> dendrites, String name, double threshold, double value)
            : base(dendrites, name, threshold)
        {
            Value = value;
            this.Dendrites = new List<Dendrite>();
        }
	
	    /**
	     * getOutput
	     * Static INeurons return their value
	     * @return double value
	     */
        public override double GetOutput(NeuralPathway path = null)
        {
		    return Value;
	    }
	
	    /**
	     * backpropagate
	     * Static neurons do not need to backpropagate
	     */
	    public override void Backpropagate(double error, NeuralPathway path = null) {}


        public string Stringify()
        {
            var s = "";
            s += "<threshold>" + Threshold + "</threshold>";
            s += "<name>" + Name + "</name>";
            s += "<value>" + Value + "</value>";

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

        public static Input Objectify(string str)
        {
            var threshold = Convert.ToDouble(Stringy.SplitOn(str, "threshold")[0]);
            var value = Convert.ToDouble(Stringy.SplitOn(str, "value")[0]);
            var name = Stringy.SplitOn(str, "name")[0];

            var ds = Stringy.SplitOn(Stringy.SplitOn(str, "dendrites")[0], "dendrite");
            var dendrites = new List<Dendrite>();
            for (var d = 0; d < ds.Count(); d++)
            {
                dendrites[d] = Dendrite.Objectify(ds[d]);
            }
            return new Input(dendrites, name, threshold, value);
        }

        public static List<List<Input>> Copy(List<List<Input>> input)
        {
            return input.Select(Copy).ToList();
        }

        public static List<Input> Copy(List<Input> input)
        {
            return input.Select(Copy).ToList();
        }

        public static Input Copy(Input input){
            return new Input(Dendrite.Copy(input.Dendrites), input.Name, input.Threshold, input.Value);
        }

        public new void AddDendrites(List<Neuron> dendrites)
        {
            //Inputs do not have dendrites
        }

    }
}
