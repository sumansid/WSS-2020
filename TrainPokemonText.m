(***********************************)
(* Hyper-parameters (to be tuned!) *)
(***********************************)

$jensen = False; (* Jensen or Wasserstein *)
$numposfeatures = 0;
$numlatent = 16;
$numhiddens = 128;
$depth = 2;
$batchsize = 32;
$discriminatorTerminalTokensQ = True;
$generatorTerminalTokensQ = True;


(* Use a folder name to dump intermediate results into it *)
$SAVEDIR = None;

$device = "CPU"; (* Can be "GPU" *)

(**********************)
(* Load Pokemon names *)
(**********************)

$DATADIR = If[$Notebooks, NotebookDirectory[], Directory[]];
fn = FileNameJoin[{$DATADIR, "poknames.mx"}];
If[FileExistsQ[fn],
	names = Import[fn];
,
	EntityPrefetch[EntityProperty["Pokemon", "Name"]];
	names = EntityValue[#, EntityProperty["Pokemon", "Name"]] & /@ EntityList["Pokemon"];
	names = DeleteDuplicates @ DeleteCases[s_String/;StringContainsQ[s,":"]] @ Map[StringTrim @ StringReplace[{
		WordBoundary~~(WordCharacter..)~~"." :> "", (* Remove "mr.", "jr.", ... *)
		"\[Mars]"|"\[Venus]" -> "", (* Remove gender hints *)
		"\[Hyphen]"|"-"|"'" -> " ", (* Remove very rare characters *)
		DigitCharacter -> "" (* Remove very rare characters *)
		}] @ First @ StringSplit[#, "("]&, names];
	Export[fn, names];
];

(***********************)
(* Text pre-processing *)
(***********************)

normalizednames = ToLowerCase @ RemoveDiacritics @ names;
counts = Counts[StringLength /@ normalizednames]; (* We will use this to generate a realistic length *)

characters = Union[Flatten @ Characters @ normalizednames];
characters2 = If[$discriminatorTerminalTokensQ,
	Join[characters, {StartOfString, EndOfString}],
	characters
];
netpreproc = NetEncoder[{"Characters", characters2, "IgnoreCase" -> True}];

(********************)
(* Net architecture *)
(********************)

nonlinearityGenerator = ElementwiseLayer[Ramp];
nonlinearityDiscriminator = ElementwiseLayer[If[# > 0, #, 0.2 * #]&];

batchnorm = BatchNormalizationLayer[(*"Scaling" -> None, "Biases" -> None*)"Scaling" -> 1, "Biases" -> 0, "Interleaving" -> True, LearningRateMultipliers -> {"Scaling" -> 0, "Biases" -> 0, _->1}];
instancenorm = NormalizationLayer[1, "Scaling" -> None, "Biases" -> None];

normalizationGenerator = batchnorm;
normalizationDiscriminator = batchnorm;

convolutionBlock[n_, args___] := NetChain[{
	ConvolutionLayer[n, {5}, "Stride" -> {1}, PaddingSize->2, "Interleaving" -> True, args],
	normalizationDiscriminator,
	nonlinearityDiscriminator,
	DropoutLayer[]
}];
deconvolutionBlock[n_, args___] := NetChain[{
	ConvolutionLayer[n, {5}, "Stride"->{1}, PaddingSize->2, "Interleaving" -> True, args],
	normalizationGenerator,
	nonlinearityGenerator,
	DropoutLayer[]
}];

textDiscriminator = NetChain[<|
	(* Preprocessing: only keep the maximum values *)
	"keep max only" -> NetGraph[{AggregationLayer[Max, -1], ThreadingLayer[If[#1 >= #2-1.*^-7,#1,0]&, -1]}, {{NetPort["Input"],1}->2}],
	Sequence @@ Table["conv."<>ToString[i] -> convolutionBlock[$numhiddens], {i, $depth}],
	(*"aggregate" -> NetGraph[{AggregationLayer[Mean,1], AggregationLayer[StandardDeviation,1],CatenateLayer[]},{{1,2}->3}],*)
	"aggregate" -> AggregationLayer[Mean,1],
	"dropout" -> DropoutLayer[],
	"classify" -> LinearLayer["Real", "Weights" -> 0, "Biases" -> None],
	If[$jensen, "logit" -> LogisticSigmoid, Nothing]
|>, "Input" -> {"Varying", Length[characters2]}];

textGenerator = NetChain[<|
	If[$generatorTerminalTokensQ,
		(* Append/Prepend EOS/SOS feature vectors *)
		"add eos/sos latent" -> NetGraph[{ArrayLayer[], AppendLayer[], ArrayLayer[], PrependLayer[]}, {{NetPort["Input"], 1} -> 2, {2, 3} -> 4}],
		Nothing],
	(* Catenate Positional embeddings *)
	If[$numposfeatures > 0,
		"pos.embedding" -> With[{nmax = Max[counts]},
			(* Inspired from "Attention Is All You Need" paper *)
			posweights = Join[
				Array[N @ Sin[#2 / Power[100, 2 * #1 / $numposfeatures]]&, {$numposfeatures/2, nmax}],
				Array[N @ Cos[#2 / Power[100, 2 * #1 / $numposfeatures]]&, {$numposfeatures/2, nmax}]
			];
			NetGraph[{
				NeuralNetworks`SequenceIndicesLayer[],
				ElementwiseLayer[Clip[#, {1, nmax}]&],
				EmbeddingLayer[$numposfeatures, nmax, "Weights" -> Transpose @ posweights
					, LearningRateMultipliers -> None (* Q: should we freeze? *)
				],
				CatenateLayer[2]
			}, {1 -> 2 -> 3, {NetPort["Input"], 3} -> 4}]
		],
		Nothing],
	(* Core deep net *)
	Sequence @@ Table["conv."<>ToString[i] -> convolutionBlock[$numhiddens], {i, $depth}],
	If[$generatorTerminalTokensQ,
		(* Remove EOS/SOS high level features*)
		"remove eos/sos prediction" -> NetChain[{
			SequenceRestLayer[],
			SequenceMostLayer[]}
		],
		Nothing],
	(* Classifier (of characters) *)
	"classify" -> NetMapOperator @ LinearLayer[Length[characters](*, "Weights" -> 0*)],
	"squash" -> LogisticSigmoid, (*SoftmaxLayer[],*)
	If[$discriminatorTerminalTokensQ,
		"add eos/sos onehot proba" -> NetGraph[{
			(* Catenate zero proba for EOS/SOS inside the generated text *)
			PaddingLayer[{{0,0}, {0,2}}],
			(* Append/Prepend proba of 1 for EOS/SOS at the end/beginning of the generated text (to be in accordance with the discriminator) *)
			ArrayLayer["Array" -> UnitVector[Length[characters2], Length[characters2]-1], LearningRateMultipliers -> None], PrependLayer[],
			ArrayLayer["Array" -> UnitVector[Length[characters2], Length[characters2]], LearningRateMultipliers -> None], AppendLayer[]},
			{{1, 2} -> 3, {3, 4} -> 5}],
		Nothing]
	|>,
	"Input" -> {"n", $numlatent},
	"Output" -> With[{netpostproc = NetDecoder[netpreproc]},
		(* Capitalize the decoded (lower-case) text *)
		NetDecoder[{"Function", Function[StringReplace[netpostproc[#], WordBoundary ~~ c:WordCharacter :> ToUpperCase[c]]]}]
	]
];

gan = NetGANOperator[{textGenerator, textDiscriminator}];

(**********************************************)
(* Generation of latent and sampling of reals *)
(**********************************************)

(* Generation of latent random *)
latentGeneration[batchSize_] := Table[
	Block[{len = Max[3, RandomChoice[Values[counts] -> Keys[counts]]]},
		NumericArray[#, "Real32"]& @ Clip[
			RandomVariate[NormalDistribution[], {len, $numlatent}],
			{-1, 1}
		]
	], batchSize
];

randomizeOnehot[onehot_] := If[$discriminatorTerminalTokensQ,
	Join[
		{First @ onehot},
		Map[# * Clip[RandomVariate[NormalDistribution[0.8, 0.1]], {0.55, 1}]&, Rest @ Most @ onehot],
		{Last @ onehot}
	],
	Map[# * Clip[RandomVariate[NormalDistribution[0.8, 0.1]], {0.55, 1}]&, onehot]
];
sampleGeneration[batchSize_] := Block[{s, onehot},
	s = RandomSample[normalizednames, batchSize];
	onehot = UnitVectorLayer[Length[characters2], "Input" -> {Automatic}][netpreproc[s]];
	NumericArray[#, "Real32"]& /@ Map[randomizeOnehot, onehot]
];

dataGenerator = {
	Function[<|"Sample" -> sampleGeneration[#BatchSize], "Latent" -> latentGeneration[#BatchSize]|>],
	"RoundLength" -> Length[normalizednames]
};


(***********************)
(* Training Monitoring *)
(***********************)

$monitoringLatent = SortBy[latentGeneration[50], Length];
monitorGAN[gan_] := Block[{generator = NetExtract[gan,"Generator"]},
	Framed @ Column[{
		Framed @ Grid[Transpose @ Partition[generator[$monitoringLatent],10], Alignment -> Left],
		Framed @ Grid[Transpose @ Partition[generator[$monitoringLatent, NetEvaluationMode -> "Train"],10], Alignment -> Left]
	}]
]

(***********************)
(* Go!!!!!!!!!!!!!!!!! *)
(***********************)

If[$Notebooks, Echo["Training of" -> gan]];

If[$SAVEDIR =!= None, CreateDirectory[FileNameJoin[{$SAVEDIR, "monitoring"}], CreateIntermediateDirectories -> True]];
trained = NetTrain[
	gan,
	dataGenerator,
	TargetDevice -> $device,
	BatchSize -> $batchsize, 
	TrainingUpdateSchedule -> {"Discriminator", "Generator"},
	MaxTrainingRounds -> 100000,
	TrainingProgressReporting -> Append[
		If[$Notebooks,
			{{Function[monitorGAN[#Net]], "Interval" -> Quantity [1, "Seconds"]}, "Panel"},
			"Print"
		],
		If[$SAVEDIR === None, Nothing, File[FileNameJoin[{$SAVEDIR, "training_log.csv"}]]]
	],
	TrainingProgressFunction -> If[$SAVEDIR === None, None,
		{
			{Function @ Block[
				{images, id},
					Check[
						id = IntegerString[#Round, 10, 7];
						example = Rasterize @ monitorGAN[#Net];
						Export[FileNameJoin[{$SAVEDIR, "monitoring/generation_"<>id<>".png"}], example];
					,
						Abort[]];
				], 
				"Interval" -> Quantity [10, "Rounds"]
			}
		}],
	TrainingProgressCheckpointing -> If[$SAVEDIR === None, None,
		{"File", $SAVEDIR<>"/gan.wlnet", "Interval" -> Quantity[10, "Rounds"]}]
];
If[FailureQ[trained], trained,
	textGenerator = NetExtract[trained, "Generator"]
]