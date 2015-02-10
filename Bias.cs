using System;

namespace ArtificialNeuralNetwork
{
    public class Bias : Input {
	    protected static double DefaultValue = -1;
	    protected static String DefaultName = "B";

	    public Bias():base(DefaultName, DefaultValue) {}
    }
}
