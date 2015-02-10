using System;
using System.Collections.Generic;
using System.Linq;
using Utilities;

namespace ArtificialNeuralNetwork
{
    public class Network
    {
        public enum TrainingAlgorithm
        {
            Normal,
            HoldBest,
            HoldBestZeroIn,
            HoldBestSpiralOut,
        }

        public static int DefaultMaxEpochs = 2000;
        public static int DefaultMaxMinima = 250;
	    public static double DefaultTargetError = 0.005;
	    public double Inputs;
	    public double Outputs;
        public int MaxEpochs = DefaultMaxEpochs;
        public int MaxMinima = DefaultMaxMinima;
        public double TargetError = DefaultTargetError;
	    public int Epochs;
	    public double Error;
        public bool WeightToRecent = false;
	    public List<Input> Neurons;
        public List<List<Dynamic>> HLayers;
        public List<Dynamic> ONeurons;

        public Network()
	    {
	    }

        public Network(int inputsNo, IReadOnlyCollection<int> hiddenNos, int outputsNo)
            : this()
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

        public Network(double numInputs, double numOutputs, int maxEpochs, double targetError,
            int epochs, double error, List<Input> neurons, List<List<Dynamic>> hLayers, List<Dynamic> oNeurons)
        {
            Inputs = numInputs;
            Outputs = numOutputs;
            MaxEpochs = maxEpochs;
            TargetError = targetError;
            Epochs = epochs;
            Error = error;
            Neurons = neurons;
            HLayers = hLayers;
            ONeurons = oNeurons;
        }
	
	    /**
	     * run
	     * Returns the outputs from the given inputs
	     * @param inputs, a List<double> of input values
	     * @return outputs, a List<double> of output values from 0-1
	     */
	    public List<double> Run(List<double> inputs) {
		    //If inputs and/or outputs aren't set up then bail now
		    if(!InputsValid(inputs) || !OutputsValid())
			    return null;
		
		    //Load input values into the input neurons
		    for(var i = 0; i < inputs.Count; i++)
			    Neurons[i].Value = inputs[i];
		
		    return Run();
	    }
	
	    /**
	     * run
	     * Returns the outputs from the stored inputs
	     * @return outputs, a List<double> of output values from 0-1
	     */
        public List<double> Run()
        {
            //If inputs and/or outputs aren't set up then bail now
		    if(!InputsValid() || !OutputsValid())
			    return null;
		
		    //Pull the output from the output neurons

            return ONeurons.Select(o => o.GetOutput()).ToList();
	    }
	
	    /**
	     * train
	     * trains the network from given inputs and target outputs
	     * @param inputs, List<List<double>> a list of input Lists
	     * @param targets, List<List<double>> a list of corresponding target Lists
	     */
        public void Train(List<List<double>> inputs, List<List<double>> targets, TrainingAlgorithm trainingAlgorithm = TrainingAlgorithm.HoldBest)
        {
            switch (trainingAlgorithm)
            {
                case (TrainingAlgorithm.Normal):
                    TrainNormal(inputs, targets);
                    break;
                case (TrainingAlgorithm.HoldBest):
                    TrainHoldBest(inputs, targets);
                    break;
                case (TrainingAlgorithm.HoldBestZeroIn):
                    TrainHoldBestZeroIn(inputs, targets);
                    break;
                case (TrainingAlgorithm.HoldBestSpiralOut):
                    TrainHoldBestSpiralOut(inputs, targets);
                    break;
                default:
                    TrainNormal(inputs, targets);
                    break;
            }
        }


	    public void TrainNormal(List<List<double>> inputs, List<List<double>> targets) 
        {
		    Epochs = 0;
		    do{
                Error = TrainEpoch(inputs, targets);
                Epochs++;
		    }while(Error > TargetError && Epochs < MaxEpochs);
	    }

        public void TrainHoldBest(List<List<double>> inputs, List<List<double>> targets)
        {
            Epochs = 0;
            var minima = 0;
            double bestError = -1;
            var bestWeights = GetWeights();
            do
            {
                Error = TrainEpoch(inputs, targets);
                Epochs++;
                minima++;
                if (Error < bestError || bestError < 0)
                {
                    minima = 0;
                    bestError = Error;
                    bestWeights = GetWeights();
                }
            } while (Error > TargetError && minima < MaxMinima && Epochs < MaxEpochs);
            SetWeights(bestWeights);
        }

        public void TrainHoldBestZeroIn(List<List<double>> inputs, List<List<double>> targets)
        {
            Epochs = 0;
            double bestError = -1;
            double previousError;
            var iterations = 0;
            var network = Copy(this);
            do
            {
                previousError = Error;
                Error = TrainEpoch(inputs, targets);
                Epochs++;
                iterations++;
                if (Error < bestError || bestError < 0)
                {
                    iterations = 0;
                    bestError = Error;
                    network = Copy(this);
                }
                else
                {
                    if (Epochs % (MaxEpochs / 100) == 0)
                    {
                        AdjustLearningRateDown();
                    }
                }
            } while ((Error < previousError) || (Error > TargetError && iterations < (MaxEpochs / 10) && Epochs < MaxEpochs) );
            Neurons = network.Neurons;
            HLayers = network.HLayers;
            ONeurons = network.ONeurons;
            PlugIn(ONeurons, HLayers, Neurons);
        }

        public void TrainHoldBestSpiralOut(List<List<double>> inputs, List<List<double>> targets)
        {
            Epochs = 0;
            var minima = 0;
            double bestError = -1;
            var bestWeights = GetWeights();
            do
            {
                Error = TrainEpoch(inputs, targets);
                Epochs++;
                minima++;
                if (Error < bestError || bestError < 0)
                {
                    minima = 0;
                    bestError = Error;
                    bestWeights = GetWeights();
                    AdjustLearningRateDown();

                }
                if(minima >= MaxMinima/2)
                {
                    AdjustLearningRateUp();
                }
            } while (Error > TargetError && minima < MaxMinima && Epochs < MaxEpochs);
            SetWeights(bestWeights);
        }

        private void AdjustLearningRateDown()
        {
            foreach (var h in HLayers.SelectMany(l => l))
            {
                h.HalveLearningRate();
            }
            foreach (var o in ONeurons)
            {
                o.HalveLearningRate();
            }
        }
        private void AdjustLearningRateUp()
        {
            foreach (var h in HLayers.SelectMany(l => l))
            {
                h.LearningRate += (1 - h.LearningRate)/2;
            }
            foreach (var o in ONeurons)
            {
                o.LearningRate += (1 - o.LearningRate) / 2;
            }
        }

        private double TrainEpoch(IReadOnlyList<List<double>> inputs, IReadOnlyList<List<double>> targets)
        {
            Error = 0;
            //for each of the training cases
            for (var i = 0; i < inputs.Count; i++)
            {
                //get the output List for the given input List
                var outputs = Run(inputs[i]);
                //For each of the output values
                for (var j = 0; j < outputs.Count; j++)
                {
                    //Find the error
                    var err = targets[i][j] - outputs[j];

                    //And backpropagate to train the network
                    ONeurons[j].Backpropagate(err);
                    //Store error for checking
                    Error = Error + Math.Abs(err);
                }
            }
            return Error;
        }

	    //Create
	
	    /**
	     * createInputs
	     * sets up the input layer
	     * @param num, number of inputs
	     * @return true if the create was successful
	     */
	    public bool CreateInputs(int num){
		    Inputs = num;
            Neurons = new List<Input>();
            for (int i = 0; i < num; i++)
			    Neurons.Add(new Input("I"+(i+1), 0));
		
		    return true;
	    }

        public static void PlugIn(List<Dynamic> outputs, List<List<Dynamic>> hiddens, List<Input> inputs)
        {
            foreach (var o in outputs)
            {
                o.PlugIn(hiddens[0]);
            }
            for (var l = 0; l < hiddens.Count - 1; l++)
            {
                foreach (var h in hiddens[l])
                {
                    h.PlugIn(hiddens[l + 1]);
                }
            }
            foreach (var h in hiddens[hiddens.Count - 1])
            {
                h.PlugIn(inputs);
            }
        }

	    /**
	     * createHiddenLayers
	     * sets up the input layer
	     * @param num, number of layers
	     * @return true if the create was successful
	     */
	    public bool CreateHiddenLayers(int num) {
            HLayers = new List<List<Dynamic>>();
		    for(int i = 0; i < num; i++)
                HLayers.Add(new List<Dynamic>());
		    return true;
	    }
	
	    /**
	     * createHiddenLayer
	     * sets up the specified layer with 'num' neurons
	     * @param layer, the hidden layer to set from 1-n
	     * @param num, number of hidden neurons in the layer
	     * @return true if the create was successful
	     * @prereq createInputs, createHiddenLayers
	     */
	    public bool CreateHiddenLayer(int layer, int num) {
		    if(!InputsValid() || layer < 1 || num < 1)
			    return false;
            HLayers[layer - 1] = new List<Dynamic>();
		    //add 'num' neurons to hidden layer 'layer'
		    for(int i = 0; i < num; i++)
			    HLayers[layer-1].Add(new Dynamic("H"+(layer-1)+"."+(i+1)));

		    //if this is the first hidden layer connect to input neurons
            if (layer == 1)
            {
                foreach (var h in HLayers[layer - 1])
                    foreach (var i in Neurons)
                        h.AddDendrite(i);
            }
		    //if this is the second+ layer, connect to previous hidden neurons
            if (layer > 1)
            {
                foreach (var h in HLayers[layer - 1])
                    foreach (var p in HLayers[layer - 2])
                        h.AddDendrite(p);
            }
		    return true;
	    }
	
	    /**
	     * createOutputs
	     * sets up the output layer with 'num' neurons
	     * @param layer, the hidden layer to set from 1-n
	     * @param num, number of hidden neurons in the layer
	     * @return true if the create was successful
	     * @prereq createInputs, createHiddenLayers, createHiddenLayer
	     */
	    public bool CreateOutputs(int num){
		    if(!InputsValid() || !HiddenLayersValid())
			    return false;
		
		    Outputs = num;
            ONeurons = new List<Dynamic>();
		
		    //Create Layer
		    for(int i = 0; i < num; i++)
			    ONeurons.Add(new Dynamic("O"+(i+1)));
		
		    //if there is at least one hidden layer, hook up to last one
            if (HiddenLayersExist())
            {
                foreach (var o in ONeurons)
                    foreach (var h in HLayers[HLayers.Count() - 1])
                        o.AddDendrite(h);
            }
            //else hook up to the input layer
            else
            {
                foreach (var o in ONeurons)
                    foreach (var i in Neurons)
                        o.AddDendrite(i);
            }
		    return true;
	    }
	
	    //Validate
	
	    /**
	     * inputsValid
	     * @return true if input layer exists
	     */
	    private bool InputsValid()
	    {
	        const double tolerance = 0.0001;
	        return !(Math.Abs(Inputs) < tolerance);
	    }

        /**
	     * inputsValid
	     * @param inputs
	     * @return true if the input layer can accept the given List
	     */
        private bool InputsValid(List<double> inputs)
        {
            const double tolerance = 0.0001;
            if (Math.Abs(Inputs) < tolerance)
			    return false;
            return !(Math.Abs(inputs.Count - Inputs) > tolerance);
        }
	
	    /**
	     * hiddenLayersExist
	     * @return true if hidden layers exist
	     */
	    private bool HiddenLayersExist(){return (HLayers.Any());}

	    /**
	     * hiddenLayersValid
	     * @return true if hidden layers are sound
	     */
	    private bool HiddenLayersValid()
	    {
	        return HLayers.All(layer => layer.All(neuron => neuron != null));
	    }

        /**
	     * outputsValid
	     * @return true if output layer exists
	     */
	    private bool OutputsValid()
	    {
            const double tolerance = 0.0001;
            return !(Math.Abs(Outputs) < tolerance);
	    }

        //IO
	    /**
	     * printNetwork
	     * converts the Network weights to a string
	     * @return human friendly string
	     */
	    public String Print()
	    {
		    //Set names (in case not already set)
		    String print = "";
		    for(int i = 0; i < Neurons.Count(); i++)
			    Neurons[i].Name = "I"+(i+1);
		    for(int i = 0; i < HLayers.Count(); i++)
			    for(int j = 0; j < HLayers[i].Count(); j++)
				    HLayers[i][j].Name = "H"+(i+1)+"."+(j+1);
		    for(int i = 0; i < ONeurons.Count(); i++)
			    ONeurons[i].Name ="O"+(i+1);
		
		    //Input layer has not connections
		    print = print + "Inputs: \n";
		
		    //For each hidden layer, print each connection for each neuron
		    for(int i = 0; i < HLayers.Count(); i++) {
			    print = print + "Hidden" + (i+1) + ": \n";
			    for(int j = 0; j < HLayers[i].Count(); j++) {
				    print = print + "\t" + HLayers[i][j].Name + ":\n";
				    for(int k = 0; k < HLayers[i][j].Dendrites.Count(); k++)
					    print = print + "\t\t(" + HLayers[i][j].Dendrites[k].Neuron.Name + ")" +
							    HLayers[i][j].Dendrites[k].Weight + "\n";
			    }
		    }
		
		    //print each connection for each neuron in the output layer
		    print = print + "Outputs: \n";
		    for(int i = 0; i < ONeurons.Count(); i++) {
			    print = print + "\t" + ONeurons[i].Name + ":\n";
			    for(int j = 0; j < ONeurons[i].Dendrites.Count(); j++)
				    print = print + "\t\t(" + ONeurons[i].Dendrites[j].Neuron.Name + ")" +
						    ONeurons[i].Dendrites[j].Weight + "\n";
		    }
		
		    return print;
	    }


        public string Stringify()
        {
            var s = "";
            s += "<numInputs>" + Inputs + "</numInputs>";
            s += "<numOutputs>" + Outputs + "</numOutputs>";
            s += "<maxEpochs>" + MaxEpochs + "</maxEpochs>";
            s += "<epochs>" + Epochs + "</epochs>";
            s += "<error>" + Error + "</error>";
            s += "<targetError>" + TargetError + "</targetError>";

            s += "<iNeurons>";
            foreach (var n in Neurons)
            {
                s += "<iNeuron>";
                s += n.Stringify();
                s += "</iNeuron>";
            }
            s += "</iNeurons>";

            s += "<hLayers>";
            foreach (var l in HLayers)
            {
                s += "<hLayer>";
                s += "<hNeurons>";
                foreach (var n in l)
                {
                    s += "<hNeuron>";
                    s += n.Stringify();
                    s += "</hNeuron>";
                }
                s += "</hNeurons>";
                s += "</hLayer>";
            }
            s += "</hLayers>";

            s += "<oNeurons>";
            foreach (var n in ONeurons)
            {
                s += "<oNeuron>";
                s += n.Stringify();
                s += "</oNeuron>";
            }
            s += "</oNeurons>";
            return s;
        }

        public static Network Objectify(string str)
        {
            var numInputs = Convert.ToInt32(Stringy.SplitOn(str, "numInputs")[0]);
            var numOutputs = Convert.ToInt32(Stringy.SplitOn(str, "numOutputs")[0]);
            var maxEpochs = Convert.ToInt32(Stringy.SplitOn(str, "maxEpochs")[0]);
            var epochs = Convert.ToInt32(Stringy.SplitOn(str, "epochs")[0]);
            var error = Convert.ToInt32(Stringy.SplitOn(str, "error")[0]);
            var targetError = Convert.ToInt32(Stringy.SplitOn(str, "targetError")[0]);

            var ins = Stringy.SplitOn(Stringy.SplitOn(str, "iNeurons")[0], "iNeuron");
            var inputs = ins.Select(Input.Objectify).ToList();

            var hls = Stringy.SplitOn(Stringy.SplitOn(str, "hLayers")[0], "hLayer");
            var hiddenLayers = hls.Select(l => Stringy.SplitOn(Stringy.SplitOn(str, "hNeurons")[0], "hNeuron")).Select(hns => hns.Select(Dynamic.Objectify).ToList()).ToList();

            var ons = Stringy.SplitOn(Stringy.SplitOn(str, "oNeurons")[0], "oNeuron");
            var outputs = ons.Select(Dynamic.Objectify).ToList();

            PlugIn(outputs, hiddenLayers, inputs);

            return new Network(numInputs, numOutputs, maxEpochs, targetError, epochs,
                error, inputs, hiddenLayers, outputs);
        }

        public static Network Copy(Network orig)
        {
            return new Network(orig.Inputs, orig.Outputs, orig.MaxEpochs, orig.TargetError, orig.Epochs, orig.Error,
                Input.Copy(orig.Neurons), Dynamic.Copy(orig.HLayers), Dynamic.Copy(orig.ONeurons));
        }

        private List<double> GetWeights()
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

        private void SetWeights(List<double> weights)
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
