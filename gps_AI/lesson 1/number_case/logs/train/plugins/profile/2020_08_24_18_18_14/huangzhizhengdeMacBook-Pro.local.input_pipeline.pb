	+���?+���?!+���?	��^�r8@��^�r8@!��^�r8@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$+���?�&1��?A���K7�?Y;�O��n�?*	     �e@2F
Iterator::ModelV-��?!w�qG�P@)B`��"۹?1x�qG\M@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMapJ+��?!_��}<@)J+��?1_��}<@:Preprocessing2S
Iterator::Model::ParallelMap���Q��?!�w�q!@)���Q��?1�w�q!@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�~j�t�x?!����/�@)�~j�t�x?1����/�@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor����MbP?!��)kʚ�?)����MbP?1��)kʚ�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 24.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2A8.7 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�&1��?�&1��?!�&1��?      ��!       "      ��!       *      ��!       2	���K7�?���K7�?!���K7�?:      ��!       B      ��!       J	;�O��n�?;�O��n�?!;�O��n�?R      ��!       Z	;�O��n�?;�O��n�?!;�O��n�?JCPU_ONLY