using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ArtificialNeuralNetwork
{
    public abstract class Network
    {

        public static int DefaultMaxEpochs = 1500;
        public static int DefaultMaxMinima = 50;
        public static double DefaultTargetError = 0.005;
        public int MaxEpochs = DefaultMaxEpochs;
        public int MaxMinima = DefaultMaxMinima;
        public double TargetError = DefaultTargetError;
        public int Epochs;
        public double Error;
        public bool WeightToRecent = false;
        public List<Input> INeurons;
        public List<List<Dynamic>> HLayers;
        public List<Dynamic> ONeurons;

        public enum TrainingAlgorithm
        {
            Normal,
            HoldBest,
            HoldBestZeroIn,
            HoldBestSpiralOut,
            HoldBestNarrowLearning,
            HoldBestInvestigate
        }

        public Network()
        {
        }

        /*public Network(double numInputs, double numOutputs, int maxEpochs, double targetError,
            int epochs, double error, List<Input> neurons, List<List<Dynamic>> hLayers, List<Dynamic> oNeurons)
        {
            Inputs = numInputs;
            Outputs = numOutputs;
            MaxEpochs = maxEpochs;
            TargetError = targetError;
            Epochs = epochs;
            Error = error;
            INeurons = neurons;
            HLayers = hLayers;
            ONeurons = oNeurons;
        }*/
	
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
			    INeurons[i].Value = inputs[i];
		
		    return Run();
	    }
	
	    /**
	     * run
	     * Returns the outputs from the stored inputs
	     * @return outputs, a List<double> of output values from 0-1
	     */
        protected abstract List<double> Run();
	
	    /**
	     * train
	     * trains the network from given inputs and target outputs
	     * @param inputs, List<List<double>> a list of input Lists
	     * @param targets, List<List<double>> a list of corresponding target Lists
	     */
        public void Train(List<List<double>> inputs, List<List<double>> targets, TrainingAlgorithm trainingAlgorithm = TrainingAlgorithm.HoldBestInvestigate)
        {
            TrainHoldBestInvestigate(inputs, targets);
            /*switch (trainingAlgorithm)
            {
                case (TrainingAlgorithm.Normal):
                    TrainNormal(inputs, targets);
                    break;
                case (TrainingAlgorithm.HoldBest):
                    TrainHoldBest(inputs, targets);
                    break;
                case (TrainingAlgorithm.HoldBestNarrowLearning):
                    TrainHoldBestNarrowLearning(inputs, targets);
                    break;
                case (TrainingAlgorithm.HoldBestInvestigate):
                    TrainHoldBestInvestigate(inputs, targets);
                    break;
                default:
                    TrainNormal(inputs, targets);
                    break;
            }*/
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


        public void TrainHoldBestNarrowLearning(List<List<double>> inputs, List<List<double>> targets)
        {

            Epochs = 0;
            var minima = 0;
            double minError = -1;
            double maxError = -1;
            double prevError = -1;

            var bestWeights = GetWeights();
            do
            {
                Error = TrainEpoch(inputs, targets);
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
            } while (Error > TargetError && minima < MaxMinima && Epochs < MaxEpochs);

            SetWeights(bestWeights);
        }

        public abstract void TrainHoldBestInvestigate(List<List<double>> inputs, List<List<double>> targets);

        protected static void RecordLog(IEnumerable<List<double>> log)
        {
            const string fileName = "Log.txt";
            var lines = log.Select(l => String.Format("{0},{1},{2},{3},{4}", l[0], l[1], l[2], l[3], l[4])).ToList();
            var file = new StreamWriter(fileName);
            foreach (var l in lines)
            {
                file.WriteLine(l);
            }
            file.Close();
        }

        protected static void RecordRankings(Dictionary<string, double> rankings)
        {

            const string fileName = "Rankings.txt";
            var file = new StreamWriter(fileName);
            foreach (var r in rankings)
            {
                file.WriteLine(r.Key + "," + r.Value);
            }
            file.Close();
        }

        protected void AdjustLearningRateDown()
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


        protected double TrainEpoch(IReadOnlyList<List<double>> inputs, IReadOnlyList<List<double>> targets)
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
                    Backpropagate(ONeurons[j], err);
                    
                    //Store error for checking
                    Error = Error + Math.Abs(err);
                }
            }
            return Error;
        }

        protected abstract void Backpropagate(Dynamic oNeuron, double err);

        //Create
	
	    /**
	     * createInputs
	     * sets up the input layer
	     * @param num, number of inputs
	     * @return true if the create was successful
	     */
	    public bool CreateInputs(int num){
            INeurons = new List<Input>();
            for (int i = 0; i < num; i++)
			    INeurons.Add(new Input("I"+(i+1), 0));
		
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
                    foreach (var i in INeurons)
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
                    foreach (var i in INeurons)
                        o.AddDendrite(i);
            }
		    return true;
	    }
	
	    //Validate
	
	    /**
	     * inputsValid
	     * @return true if input layer exists
	     */
        protected bool InputsValid()
	    {
	        return INeurons.Count > 0;
	    }

        /**
	     * inputsValid
	     * @param inputs
	     * @return true if the input layer can accept the given List
	     */
        protected bool InputsValid(List<double> inputs)
        {
            return inputs.Count == INeurons.Count;
        }
	
	    /**
	     * hiddenLayersExist
	     * @return true if hidden layers exist
	     */
        protected bool HiddenLayersExist() { return (HLayers.Any()); }

	    /**
	     * hiddenLayersValid
	     * @return true if hidden layers are sound
	     */
        protected bool HiddenLayersValid()
	    {
	        return HLayers.All(layer => layer.All(neuron => neuron != null));
	    }

        /**
	     * outputsValid
	     * @return true if output layer exists
	     */

        protected bool OutputsValid()
	    {
            return ONeurons.Count > 0;
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
		    for(int i = 0; i < INeurons.Count(); i++)
			    INeurons[i].Name = "I"+(i+1);
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

        /*
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
            foreach (var n in INeurons)
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
        */
        /*public static Network Objectify(string str)
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
                Input.Copy(orig.INeurons), Dynamic.Copy(orig.HLayers), Dynamic.Copy(orig.ONeurons));
        }*/

        public abstract List<double> GetWeights();

        public abstract void SetWeights(List<double> weights);

        public List<NeuralPathway> GetPathways()
        {
            var pathways = new List<NeuralPathway>();
            foreach (var o in ONeurons)
            {
                var path = new NeuralPathway();
                path.Path.Add(o);
                path.Weightings.Add(1);
                var paths = o.ExtendPathway(path);
                pathways.AddRange(paths);
            }
            return pathways;
        }

        public Dictionary<string, double> RankInputs()
        {
            var rankings = new Dictionary<string, double>();
            foreach (var i in INeurons)
            {
                foreach (var p in GetPathways().Where(p => p.Path.Any(n => String.Equals(n.Name, i.Name, StringComparison.CurrentCultureIgnoreCase))))
                {
                    if (rankings.ContainsKey(i.Name))
                    {
                        if (rankings[i.Name] > p.WeightingProduct())
                        {
                            rankings[i.Name] = p.WeightingProduct();
                        }
                    }
                    else
                    {
                        rankings.Add(i.Name, p.WeightingProduct());
                    }
                    
                }
            }
            return rankings;
        }
    }
}
