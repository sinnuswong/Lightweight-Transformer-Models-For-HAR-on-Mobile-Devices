��	
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02unknown8׭
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
|
Adam/v/Output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/Output/bias
u
&Adam/v/Output/bias/Read/ReadVariableOpReadVariableOpAdam/v/Output/bias*
_output_shapes
:*
dtype0
|
Adam/m/Output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/m/Output/bias
u
&Adam/m/Output/bias/Read/ReadVariableOpReadVariableOpAdam/m/Output/bias*
_output_shapes
:*
dtype0
�
Adam/v/Output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/v/Output/kernel
}
(Adam/v/Output/kernel/Read/ReadVariableOpReadVariableOpAdam/v/Output/kernel*
_output_shapes

:*
dtype0
�
Adam/m/Output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/m/Output/kernel
}
(Adam/m/Output/kernel/Read/ReadVariableOpReadVariableOpAdam/m/Output/kernel*
_output_shapes

:*
dtype0
~
Adam/v/Dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/Dense_2/bias
w
'Adam/v/Dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/Dense_2/bias*
_output_shapes
:*
dtype0
~
Adam/m/Dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/Dense_2/bias
w
'Adam/m/Dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/Dense_2/bias*
_output_shapes
:*
dtype0
�
Adam/v/Dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/v/Dense_2/kernel

)Adam/v/Dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/Dense_2/kernel*
_output_shapes

:*
dtype0
�
Adam/m/Dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/m/Dense_2/kernel

)Adam/m/Dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/Dense_2/kernel*
_output_shapes

:*
dtype0
~
Adam/v/Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/Dense_1/bias
w
'Adam/v/Dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/Dense_1/bias*
_output_shapes
:*
dtype0
~
Adam/m/Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/Dense_1/bias
w
'Adam/m/Dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/Dense_1/bias*
_output_shapes
:*
dtype0
�
Adam/v/Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/v/Dense_1/kernel
�
)Adam/v/Dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/Dense_1/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/m/Dense_1/kernel
�
)Adam/m/Dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/Dense_1/kernel*
_output_shapes
:	�*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
n
Output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameOutput/bias
g
Output/bias/Read/ReadVariableOpReadVariableOpOutput/bias*
_output_shapes
:*
dtype0
v
Output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameOutput/kernel
o
!Output/kernel/Read/ReadVariableOpReadVariableOpOutput/kernel*
_output_shapes

:*
dtype0
p
Dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameDense_2/bias
i
 Dense_2/bias/Read/ReadVariableOpReadVariableOpDense_2/bias*
_output_shapes
:*
dtype0
x
Dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameDense_2/kernel
q
"Dense_2/kernel/Read/ReadVariableOpReadVariableOpDense_2/kernel*
_output_shapes

:*
dtype0
p
Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameDense_1/bias
i
 Dense_1/bias/Read/ReadVariableOpReadVariableOpDense_1/bias*
_output_shapes
:*
dtype0
y
Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_nameDense_1/kernel
r
"Dense_1/kernel/Read/ReadVariableOpReadVariableOpDense_1/kernel*
_output_shapes
:	�*
dtype0
j
serving_default_xPlaceholder*"
_output_shapes
:*
dtype0*
shape:
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_xDense_1/kernelDense_1/biasDense_2/kernelDense_2/biasOutput/kernelOutput/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_34960

NoOpNoOp
�1
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�0
value�0B�0 B�0
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias*
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias*
.
0
1
"2
#3
*4
+5*
.
0
1
"2
#3
*4
+5*
,
,0
-1
.2
/3
04
15* 
�
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
7trace_0
8trace_1
9trace_2
:trace_3* 
6
;trace_0
<trace_1
=trace_2
>trace_3* 
* 
�
?
_variables
@_iterations
A_learning_rate
B_index_dict
C
_momentums
D_velocities
E_update_step_xla*

Fserving_default* 
* 
* 
* 
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Ltrace_0* 

Mtrace_0* 

0
1*

0
1*

,0
-1* 
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Strace_0* 

Ttrace_0* 
^X
VARIABLE_VALUEDense_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEDense_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

"0
#1*

"0
#1*

.0
/1* 
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

Ztrace_0* 

[trace_0* 
^X
VARIABLE_VALUEDense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEDense_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

*0
+1*

*0
+1*

00
11* 
�
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

atrace_0* 

btrace_0* 
]W
VARIABLE_VALUEOutput/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEOutput/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

ctrace_0* 

dtrace_0* 

etrace_0* 

ftrace_0* 

gtrace_0* 

htrace_0* 
* 
 
0
1
2
3*

i0
j1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
b
@0
k1
l2
m3
n4
o5
p6
q7
r8
s9
t10
u11
v12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
k0
m1
o2
q3
s4
u5*
.
l0
n1
p2
r3
t4
v5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

,0
-1* 
* 
* 
* 
* 
* 
* 

.0
/1* 
* 
* 
* 
* 
* 
* 

00
11* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
w	variables
x	keras_api
	ytotal
	zcount*
H
{	variables
|	keras_api
	}total
	~count

_fn_kwargs*
`Z
VARIABLE_VALUEAdam/m/Dense_1/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/Dense_1/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/Dense_1/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/Dense_1/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/Dense_2/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/Dense_2/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/Dense_2/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/Dense_2/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/Output/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/Output/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/Output/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/Output/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*

y0
z1*

w	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

}0
~1*

{	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameDense_1/kernelDense_1/biasDense_2/kernelDense_2/biasOutput/kernelOutput/bias	iterationlearning_rateAdam/m/Dense_1/kernelAdam/v/Dense_1/kernelAdam/m/Dense_1/biasAdam/v/Dense_1/biasAdam/m/Dense_2/kernelAdam/v/Dense_2/kernelAdam/m/Dense_2/biasAdam/v/Dense_2/biasAdam/m/Output/kernelAdam/v/Output/kernelAdam/m/Output/biasAdam/v/Output/biastotal_1count_1totalcountConst*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_35839
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameDense_1/kernelDense_1/biasDense_2/kernelDense_2/biasOutput/kernelOutput/bias	iterationlearning_rateAdam/m/Dense_1/kernelAdam/v/Dense_1/kernelAdam/m/Dense_1/biasAdam/v/Dense_1/biasAdam/m/Dense_2/kernelAdam/v/Dense_2/kernelAdam/m/Dense_2/biasAdam/v/Dense_2/biasAdam/m/Output/kernelAdam/v/Output/kernelAdam/m/Output/biasAdam/v/Output/biastotal_1count_1totalcount*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_35921��
�5
�
G__inference_sequential_1_layer_call_and_return_conditional_losses_35144
flatten_input 
dense_1_35104:	�
dense_1_35106:
dense_2_35109:
dense_2_35111:
output_35114:
output_35116:
identity��Dense_1/StatefulPartitionedCall�.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp�0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�Dense_2/StatefulPartitionedCall�.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp�0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�Output/StatefulPartitionedCall�-Output/bias/Regularizer/L2Loss/ReadVariableOp�/Output/kernel/Regularizer/L2Loss/ReadVariableOp�
flatten/PartitionedCallPartitionedCallflatten_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_34998�
Dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_35104dense_1_35106*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_35019�
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_35109dense_2_35111*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_35044�
Output/StatefulPartitionedCallStatefulPartitionedCall(Dense_2/StatefulPartitionedCall:output:0output_35114output_35116*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Output_layer_call_and_return_conditional_losses_35069
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_35104*
_output_shapes
:	�*
dtype0�
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0*Dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_35106*
_output_shapes
:*
dtype0�
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0(Dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ~
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_35109*
_output_shapes

:*
dtype0�
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/kernel/Regularizer/mulMul)Dense_2/kernel/Regularizer/mul/x:output:0*Dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_35111*
_output_shapes
:*
dtype0�
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/bias/Regularizer/mulMul'Dense_2/bias/Regularizer/mul/x:output:0(Dense_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_35114*
_output_shapes

:*
dtype0�
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_35116*
_output_shapes
:*
dtype0�
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/bias/Regularizer/mulMul&Output/bias/Regularizer/mul/x:output:0'Output/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^Dense_1/StatefulPartitionedCall/^Dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^Dense_2/StatefulPartitionedCall/^Dense_2/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp^Output/StatefulPartitionedCall.^Output/bias/Regularizer/L2Loss/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : 2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2`
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2`
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2^
-Output/bias/Regularizer/L2Loss/ReadVariableOp-Output/bias/Regularizer/L2Loss/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp:Z V
+
_output_shapes
:���������
'
_user_specified_nameflatten_input
�
�
B__inference_Dense_1_layer_call_and_return_conditional_losses_35019

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp�0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0*Dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0(Dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp/^Dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2`
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
,__inference_sequential_1_layer_call_fn_35267
flatten_input
unknown:	�
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_35252o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_nameflatten_input
�
�
'__inference_Dense_2_layer_call_fn_35571

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_35044o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_5_35672D
6output_bias_regularizer_l2loss_readvariableop_resource:
identity��-Output/bias/Regularizer/L2Loss/ReadVariableOp�
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp6output_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
:*
dtype0�
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/bias/Regularizer/mulMul&Output/bias/Regularizer/mul/x:output:0'Output/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ]
IdentityIdentityOutput/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: v
NoOpNoOp.^Output/bias/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-Output/bias/Regularizer/L2Loss/ReadVariableOp-Output/bias/Regularizer/L2Loss/ReadVariableOp
�
�
'__inference_Dense_1_layer_call_fn_35543

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_35019o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_Output_layer_call_and_return_conditional_losses_35618

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�-Output/bias/Regularizer/L2Loss/ReadVariableOp�/Output/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:����������
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/bias/Regularizer/mulMul&Output/bias/Regularizer/mul/x:output:0'Output/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^Output/bias/Regularizer/L2Loss/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-Output/bias/Regularizer/L2Loss/ReadVariableOp-Output/bias/Regularizer/L2Loss/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�5
�
G__inference_sequential_1_layer_call_and_return_conditional_losses_35252

inputs 
dense_1_35212:	�
dense_1_35214:
dense_2_35217:
dense_2_35219:
output_35222:
output_35224:
identity��Dense_1/StatefulPartitionedCall�.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp�0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�Dense_2/StatefulPartitionedCall�.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp�0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�Output/StatefulPartitionedCall�-Output/bias/Regularizer/L2Loss/ReadVariableOp�/Output/kernel/Regularizer/L2Loss/ReadVariableOp�
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_34998�
Dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_35212dense_1_35214*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_35019�
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_35217dense_2_35219*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_35044�
Output/StatefulPartitionedCallStatefulPartitionedCall(Dense_2/StatefulPartitionedCall:output:0output_35222output_35224*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Output_layer_call_and_return_conditional_losses_35069
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_35212*
_output_shapes
:	�*
dtype0�
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0*Dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_35214*
_output_shapes
:*
dtype0�
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0(Dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ~
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_35217*
_output_shapes

:*
dtype0�
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/kernel/Regularizer/mulMul)Dense_2/kernel/Regularizer/mul/x:output:0*Dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_35219*
_output_shapes
:*
dtype0�
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/bias/Regularizer/mulMul'Dense_2/bias/Regularizer/mul/x:output:0(Dense_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_35222*
_output_shapes

:*
dtype0�
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_35224*
_output_shapes
:*
dtype0�
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/bias/Regularizer/mulMul&Output/bias/Regularizer/mul/x:output:0'Output/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^Dense_1/StatefulPartitionedCall/^Dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^Dense_2/StatefulPartitionedCall/^Dense_2/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp^Output/StatefulPartitionedCall.^Output/bias/Regularizer/L2Loss/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : 2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2`
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2`
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2^
-Output/bias/Regularizer/L2Loss/ReadVariableOp-Output/bias/Regularizer/L2Loss/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
 __inference__wrapped_model_34988
flatten_inputF
3sequential_1_dense_1_matmul_readvariableop_resource:	�B
4sequential_1_dense_1_biasadd_readvariableop_resource:E
3sequential_1_dense_2_matmul_readvariableop_resource:B
4sequential_1_dense_2_biasadd_readvariableop_resource:D
2sequential_1_output_matmul_readvariableop_resource:A
3sequential_1_output_biasadd_readvariableop_resource:
identity��+sequential_1/Dense_1/BiasAdd/ReadVariableOp�*sequential_1/Dense_1/MatMul/ReadVariableOp�+sequential_1/Dense_2/BiasAdd/ReadVariableOp�*sequential_1/Dense_2/MatMul/ReadVariableOp�*sequential_1/Output/BiasAdd/ReadVariableOp�)sequential_1/Output/MatMul/ReadVariableOpk
sequential_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
sequential_1/flatten/ReshapeReshapeflatten_input#sequential_1/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
*sequential_1/Dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_1/Dense_1/MatMulMatMul%sequential_1/flatten/Reshape:output:02sequential_1/Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+sequential_1/Dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/Dense_1/BiasAddBiasAdd%sequential_1/Dense_1/MatMul:product:03sequential_1/Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
sequential_1/Dense_1/ReluRelu%sequential_1/Dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
*sequential_1/Dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential_1/Dense_2/MatMulMatMul'sequential_1/Dense_1/Relu:activations:02sequential_1/Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+sequential_1/Dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/Dense_2/BiasAddBiasAdd%sequential_1/Dense_2/MatMul:product:03sequential_1/Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
sequential_1/Dense_2/ReluRelu%sequential_1/Dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:����������
)sequential_1/Output/MatMul/ReadVariableOpReadVariableOp2sequential_1_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential_1/Output/MatMulMatMul'sequential_1/Dense_2/Relu:activations:01sequential_1/Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
*sequential_1/Output/BiasAdd/ReadVariableOpReadVariableOp3sequential_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/Output/BiasAddBiasAdd$sequential_1/Output/MatMul:product:02sequential_1/Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
sequential_1/Output/SoftmaxSoftmax$sequential_1/Output/BiasAdd:output:0*
T0*'
_output_shapes
:���������t
IdentityIdentity%sequential_1/Output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^sequential_1/Dense_1/BiasAdd/ReadVariableOp+^sequential_1/Dense_1/MatMul/ReadVariableOp,^sequential_1/Dense_2/BiasAdd/ReadVariableOp+^sequential_1/Dense_2/MatMul/ReadVariableOp+^sequential_1/Output/BiasAdd/ReadVariableOp*^sequential_1/Output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : 2Z
+sequential_1/Dense_1/BiasAdd/ReadVariableOp+sequential_1/Dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/Dense_1/MatMul/ReadVariableOp*sequential_1/Dense_1/MatMul/ReadVariableOp2Z
+sequential_1/Dense_2/BiasAdd/ReadVariableOp+sequential_1/Dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/Dense_2/MatMul/ReadVariableOp*sequential_1/Dense_2/MatMul/ReadVariableOp2X
*sequential_1/Output/BiasAdd/ReadVariableOp*sequential_1/Output/BiasAdd/ReadVariableOp2V
)sequential_1/Output/MatMul/ReadVariableOp)sequential_1/Output/MatMul/ReadVariableOp:Z V
+
_output_shapes
:���������
'
_user_specified_nameflatten_input
�
�
__inference_loss_fn_1_35636E
7dense_1_bias_regularizer_l2loss_readvariableop_resource:
identity��.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp�
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp7dense_1_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
:*
dtype0�
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0(Dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ^
IdentityIdentity Dense_1/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^Dense_1/bias/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp
�
�
A__inference_Output_layer_call_and_return_conditional_losses_35069

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�-Output/bias/Regularizer/L2Loss/ReadVariableOp�/Output/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:����������
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/bias/Regularizer/mulMul&Output/bias/Regularizer/mul/x:output:0'Output/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^Output/bias/Regularizer/L2Loss/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-Output/bias/Regularizer/L2Loss/ReadVariableOp-Output/bias/Regularizer/L2Loss/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�#
�
__inference_<lambda>_34941
xF
3sequential_1_dense_1_matmul_readvariableop_resource:	�B
4sequential_1_dense_1_biasadd_readvariableop_resource:E
3sequential_1_dense_2_matmul_readvariableop_resource:B
4sequential_1_dense_2_biasadd_readvariableop_resource:D
2sequential_1_output_matmul_readvariableop_resource:A
3sequential_1_output_biasadd_readvariableop_resource:
identity��+sequential_1/Dense_1/BiasAdd/ReadVariableOp�*sequential_1/Dense_1/MatMul/ReadVariableOp�+sequential_1/Dense_2/BiasAdd/ReadVariableOp�*sequential_1/Dense_2/MatMul/ReadVariableOp�*sequential_1/Output/BiasAdd/ReadVariableOp�)sequential_1/Output/MatMul/ReadVariableOpk
sequential_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   y
sequential_1/flatten/ReshapeReshapex#sequential_1/flatten/Const:output:0*
T0*
_output_shapes
:	��
*sequential_1/Dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_1/Dense_1/MatMulMatMul%sequential_1/flatten/Reshape:output:02sequential_1/Dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
+sequential_1/Dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/Dense_1/BiasAddBiasAdd%sequential_1/Dense_1/MatMul:product:03sequential_1/Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:q
sequential_1/Dense_1/ReluRelu%sequential_1/Dense_1/BiasAdd:output:0*
T0*
_output_shapes

:�
*sequential_1/Dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential_1/Dense_2/MatMulMatMul'sequential_1/Dense_1/Relu:activations:02sequential_1/Dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
+sequential_1/Dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/Dense_2/BiasAddBiasAdd%sequential_1/Dense_2/MatMul:product:03sequential_1/Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:q
sequential_1/Dense_2/ReluRelu%sequential_1/Dense_2/BiasAdd:output:0*
T0*
_output_shapes

:�
)sequential_1/Output/MatMul/ReadVariableOpReadVariableOp2sequential_1_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential_1/Output/MatMulMatMul'sequential_1/Dense_2/Relu:activations:01sequential_1/Output/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
*sequential_1/Output/BiasAdd/ReadVariableOpReadVariableOp3sequential_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/Output/BiasAddBiasAdd$sequential_1/Output/MatMul:product:02sequential_1/Output/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:u
sequential_1/Output/SoftmaxSoftmax$sequential_1/Output/BiasAdd:output:0*
T0*
_output_shapes

:k
IdentityIdentity%sequential_1/Output/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp,^sequential_1/Dense_1/BiasAdd/ReadVariableOp+^sequential_1/Dense_1/MatMul/ReadVariableOp,^sequential_1/Dense_2/BiasAdd/ReadVariableOp+^sequential_1/Dense_2/MatMul/ReadVariableOp+^sequential_1/Output/BiasAdd/ReadVariableOp*^sequential_1/Output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 2Z
+sequential_1/Dense_1/BiasAdd/ReadVariableOp+sequential_1/Dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/Dense_1/MatMul/ReadVariableOp*sequential_1/Dense_1/MatMul/ReadVariableOp2Z
+sequential_1/Dense_2/BiasAdd/ReadVariableOp+sequential_1/Dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/Dense_2/MatMul/ReadVariableOp*sequential_1/Dense_2/MatMul/ReadVariableOp2X
*sequential_1/Output/BiasAdd/ReadVariableOp*sequential_1/Output/BiasAdd/ReadVariableOp2V
)sequential_1/Output/MatMul/ReadVariableOp)sequential_1/Output/MatMul/ReadVariableOp:E A
"
_output_shapes
:

_user_specified_namex
�
C
'__inference_flatten_layer_call_fn_35528

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_34998a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
,__inference_sequential_1_layer_call_fn_35206
flatten_input
unknown:	�
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_35191o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_nameflatten_input
�	
�
__inference_loss_fn_2_35645K
9dense_2_kernel_regularizer_l2loss_readvariableop_resource:
identity��0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp9dense_2_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0�
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/kernel/Regularizer/mulMul)Dense_2/kernel/Regularizer/mul/x:output:0*Dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentity"Dense_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: y
NoOpNoOp1^Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp
�@
�
G__inference_sequential_1_layer_call_and_return_conditional_losses_35472

inputs9
&dense_1_matmul_readvariableop_resource:	�5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity��Dense_1/BiasAdd/ReadVariableOp�Dense_1/MatMul/ReadVariableOp�.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp�0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�Dense_2/BiasAdd/ReadVariableOp�Dense_2/MatMul/ReadVariableOp�.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp�0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�Output/BiasAdd/ReadVariableOp�Output/MatMul/ReadVariableOp�-Output/bias/Regularizer/L2Loss/ReadVariableOp�/Output/kernel/Regularizer/L2Loss/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   m
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:�����������
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
Dense_1/MatMulMatMulflatten/Reshape:output:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
Dense_2/MatMulMatMulDense_1/Relu:activations:0%Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Dense_2/BiasAddBiasAddDense_2/MatMul:product:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
Dense_2/ReluReluDense_2/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
Output/MatMulMatMulDense_2/Relu:activations:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*'
_output_shapes
:����������
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0*Dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0(Dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/kernel/Regularizer/mulMul)Dense_2/kernel/Regularizer/mul/x:output:0*Dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/bias/Regularizer/mulMul'Dense_2/bias/Regularizer/mul/x:output:0(Dense_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/bias/Regularizer/mulMul&Output/bias/Regularizer/mul/x:output:0'Output/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityOutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp/^Dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp^Dense_2/MatMul/ReadVariableOp/^Dense_2/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp.^Output/bias/Regularizer/L2Loss/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : 2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2>
Dense_1/MatMul/ReadVariableOpDense_1/MatMul/ReadVariableOp2`
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2@
Dense_2/BiasAdd/ReadVariableOpDense_2/BiasAdd/ReadVariableOp2>
Dense_2/MatMul/ReadVariableOpDense_2/MatMul/ReadVariableOp2`
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp2^
-Output/bias/Regularizer/L2Loss/ReadVariableOp-Output/bias/Regularizer/L2Loss/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�f
�
!__inference__traced_restore_35921
file_prefix2
assignvariableop_dense_1_kernel:	�-
assignvariableop_1_dense_1_bias:3
!assignvariableop_2_dense_2_kernel:-
assignvariableop_3_dense_2_bias:2
 assignvariableop_4_output_kernel:,
assignvariableop_5_output_bias:&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: ;
(assignvariableop_8_adam_m_dense_1_kernel:	�;
(assignvariableop_9_adam_v_dense_1_kernel:	�5
'assignvariableop_10_adam_m_dense_1_bias:5
'assignvariableop_11_adam_v_dense_1_bias:;
)assignvariableop_12_adam_m_dense_2_kernel:;
)assignvariableop_13_adam_v_dense_2_kernel:5
'assignvariableop_14_adam_m_dense_2_bias:5
'assignvariableop_15_adam_v_dense_2_bias::
(assignvariableop_16_adam_m_output_kernel::
(assignvariableop_17_adam_v_output_kernel:4
&assignvariableop_18_adam_m_output_bias:4
&assignvariableop_19_adam_v_output_bias:%
assignvariableop_20_total_1: %
assignvariableop_21_count_1: #
assignvariableop_22_total: #
assignvariableop_23_count: 
identity_25��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_2_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_2_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp assignvariableop_4_output_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_output_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp(assignvariableop_8_adam_m_dense_1_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp(assignvariableop_9_adam_v_dense_1_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp'assignvariableop_10_adam_m_dense_1_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp'assignvariableop_11_adam_v_dense_1_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp)assignvariableop_12_adam_m_dense_2_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_v_dense_2_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_m_dense_2_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_v_dense_2_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_m_output_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_v_output_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_m_output_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp&assignvariableop_19_adam_v_output_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_35534

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_Dense_1_layer_call_and_return_conditional_losses_35562

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp�0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0*Dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0(Dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp/^Dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2`
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_sequential_1_layer_call_fn_35421

inputs
unknown:	�
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_35252o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_Dense_2_layer_call_and_return_conditional_losses_35044

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp�0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/kernel/Regularizer/mulMul)Dense_2/kernel/Regularizer/mul/x:output:0*Dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/bias/Regularizer/mulMul'Dense_2/bias/Regularizer/mul/x:output:0(Dense_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp/^Dense_2/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2`
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_4_35663J
8output_kernel_regularizer_l2loss_readvariableop_resource:
identity��/Output/kernel/Regularizer/L2Loss/ReadVariableOp�
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp8output_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0�
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentity!Output/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_34998

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�5
�
G__inference_sequential_1_layer_call_and_return_conditional_losses_35100
flatten_input 
dense_1_35020:	�
dense_1_35022:
dense_2_35045:
dense_2_35047:
output_35070:
output_35072:
identity��Dense_1/StatefulPartitionedCall�.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp�0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�Dense_2/StatefulPartitionedCall�.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp�0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�Output/StatefulPartitionedCall�-Output/bias/Regularizer/L2Loss/ReadVariableOp�/Output/kernel/Regularizer/L2Loss/ReadVariableOp�
flatten/PartitionedCallPartitionedCallflatten_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_34998�
Dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_35020dense_1_35022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_35019�
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_35045dense_2_35047*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_35044�
Output/StatefulPartitionedCallStatefulPartitionedCall(Dense_2/StatefulPartitionedCall:output:0output_35070output_35072*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Output_layer_call_and_return_conditional_losses_35069
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_35020*
_output_shapes
:	�*
dtype0�
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0*Dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_35022*
_output_shapes
:*
dtype0�
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0(Dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ~
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_35045*
_output_shapes

:*
dtype0�
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/kernel/Regularizer/mulMul)Dense_2/kernel/Regularizer/mul/x:output:0*Dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_35047*
_output_shapes
:*
dtype0�
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/bias/Regularizer/mulMul'Dense_2/bias/Regularizer/mul/x:output:0(Dense_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_35070*
_output_shapes

:*
dtype0�
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_35072*
_output_shapes
:*
dtype0�
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/bias/Regularizer/mulMul&Output/bias/Regularizer/mul/x:output:0'Output/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^Dense_1/StatefulPartitionedCall/^Dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^Dense_2/StatefulPartitionedCall/^Dense_2/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp^Output/StatefulPartitionedCall.^Output/bias/Regularizer/L2Loss/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : 2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2`
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2`
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2^
-Output/bias/Regularizer/L2Loss/ReadVariableOp-Output/bias/Regularizer/L2Loss/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp:Z V
+
_output_shapes
:���������
'
_user_specified_nameflatten_input
��
�
__inference__traced_save_35839
file_prefix8
%read_disablecopyonread_dense_1_kernel:	�3
%read_1_disablecopyonread_dense_1_bias:9
'read_2_disablecopyonread_dense_2_kernel:3
%read_3_disablecopyonread_dense_2_bias:8
&read_4_disablecopyonread_output_kernel:2
$read_5_disablecopyonread_output_bias:,
"read_6_disablecopyonread_iteration:	 0
&read_7_disablecopyonread_learning_rate: A
.read_8_disablecopyonread_adam_m_dense_1_kernel:	�A
.read_9_disablecopyonread_adam_v_dense_1_kernel:	�;
-read_10_disablecopyonread_adam_m_dense_1_bias:;
-read_11_disablecopyonread_adam_v_dense_1_bias:A
/read_12_disablecopyonread_adam_m_dense_2_kernel:A
/read_13_disablecopyonread_adam_v_dense_2_kernel:;
-read_14_disablecopyonread_adam_m_dense_2_bias:;
-read_15_disablecopyonread_adam_v_dense_2_bias:@
.read_16_disablecopyonread_adam_m_output_kernel:@
.read_17_disablecopyonread_adam_v_output_kernel::
,read_18_disablecopyonread_adam_m_output_bias::
,read_19_disablecopyonread_adam_v_output_bias:+
!read_20_disablecopyonread_total_1: +
!read_21_disablecopyonread_count_1: )
read_22_disablecopyonread_total: )
read_23_disablecopyonread_count: 
savev2_const
identity_49��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: w
Read/DisableCopyOnReadDisableCopyOnRead%read_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp%read_disablecopyonread_dense_1_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	�y
Read_1/DisableCopyOnReadDisableCopyOnRead%read_1_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp%read_1_disablecopyonread_dense_1_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:{
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_2_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_2_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:z
Read_4/DisableCopyOnReadDisableCopyOnRead&read_4_disablecopyonread_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp&read_4_disablecopyonread_output_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:x
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_output_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp$read_5_disablecopyonread_output_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_6/DisableCopyOnReadDisableCopyOnRead"read_6_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp"read_6_disablecopyonread_iteration^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_learning_rate^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_8/DisableCopyOnReadDisableCopyOnRead.read_8_disablecopyonread_adam_m_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp.read_8_disablecopyonread_adam_m_dense_1_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_9/DisableCopyOnReadDisableCopyOnRead.read_9_disablecopyonread_adam_v_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp.read_9_disablecopyonread_adam_v_dense_1_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_10/DisableCopyOnReadDisableCopyOnRead-read_10_disablecopyonread_adam_m_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp-read_10_disablecopyonread_adam_m_dense_1_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_11/DisableCopyOnReadDisableCopyOnRead-read_11_disablecopyonread_adam_v_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp-read_11_disablecopyonread_adam_v_dense_1_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_12/DisableCopyOnReadDisableCopyOnRead/read_12_disablecopyonread_adam_m_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp/read_12_disablecopyonread_adam_m_dense_2_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_13/DisableCopyOnReadDisableCopyOnRead/read_13_disablecopyonread_adam_v_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp/read_13_disablecopyonread_adam_v_dense_2_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_14/DisableCopyOnReadDisableCopyOnRead-read_14_disablecopyonread_adam_m_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp-read_14_disablecopyonread_adam_m_dense_2_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_15/DisableCopyOnReadDisableCopyOnRead-read_15_disablecopyonread_adam_v_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp-read_15_disablecopyonread_adam_v_dense_2_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_16/DisableCopyOnReadDisableCopyOnRead.read_16_disablecopyonread_adam_m_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp.read_16_disablecopyonread_adam_m_output_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_17/DisableCopyOnReadDisableCopyOnRead.read_17_disablecopyonread_adam_v_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp.read_17_disablecopyonread_adam_v_output_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_18/DisableCopyOnReadDisableCopyOnRead,read_18_disablecopyonread_adam_m_output_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp,read_18_disablecopyonread_adam_m_output_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_19/DisableCopyOnReadDisableCopyOnRead,read_19_disablecopyonread_adam_v_output_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp,read_19_disablecopyonread_adam_v_output_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_20/DisableCopyOnReadDisableCopyOnRead!read_20_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp!read_20_disablecopyonread_total_1^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_21/DisableCopyOnReadDisableCopyOnRead!read_21_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp!read_21_disablecopyonread_count_1^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_22/DisableCopyOnReadDisableCopyOnReadread_22_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOpread_22_disablecopyonread_total^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_23/DisableCopyOnReadDisableCopyOnReadread_23_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOpread_23_disablecopyonread_count^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: �

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *'
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_48Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_49IdentityIdentity_48:output:0^NoOp*
T0*
_output_shapes
: �

NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_49Identity_49:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
�	
�
__inference_loss_fn_0_35627L
9dense_1_kernel_regularizer_l2loss_readvariableop_resource:	�
identity��0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp9dense_1_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	�*
dtype0�
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0*Dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentity"Dense_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: y
NoOpNoOp1^Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp
�
�
,__inference_sequential_1_layer_call_fn_35404

inputs
unknown:	�
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_35191o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�@
�
G__inference_sequential_1_layer_call_and_return_conditional_losses_35523

inputs9
&dense_1_matmul_readvariableop_resource:	�5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity��Dense_1/BiasAdd/ReadVariableOp�Dense_1/MatMul/ReadVariableOp�.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp�0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�Dense_2/BiasAdd/ReadVariableOp�Dense_2/MatMul/ReadVariableOp�.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp�0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�Output/BiasAdd/ReadVariableOp�Output/MatMul/ReadVariableOp�-Output/bias/Regularizer/L2Loss/ReadVariableOp�/Output/kernel/Regularizer/L2Loss/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   m
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:�����������
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
Dense_1/MatMulMatMulflatten/Reshape:output:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
Dense_2/MatMulMatMulDense_1/Relu:activations:0%Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Dense_2/BiasAddBiasAddDense_2/MatMul:product:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
Dense_2/ReluReluDense_2/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
Output/MatMulMatMulDense_2/Relu:activations:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*'
_output_shapes
:����������
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0*Dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0(Dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/kernel/Regularizer/mulMul)Dense_2/kernel/Regularizer/mul/x:output:0*Dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/bias/Regularizer/mulMul'Dense_2/bias/Regularizer/mul/x:output:0(Dense_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/bias/Regularizer/mulMul&Output/bias/Regularizer/mul/x:output:0'Output/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityOutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp/^Dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp^Dense_2/MatMul/ReadVariableOp/^Dense_2/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp.^Output/bias/Regularizer/L2Loss/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : 2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2>
Dense_1/MatMul/ReadVariableOpDense_1/MatMul/ReadVariableOp2`
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2@
Dense_2/BiasAdd/ReadVariableOpDense_2/BiasAdd/ReadVariableOp2>
Dense_2/MatMul/ReadVariableOpDense_2/MatMul/ReadVariableOp2`
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp2^
-Output/bias/Regularizer/L2Loss/ReadVariableOp-Output/bias/Regularizer/L2Loss/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_Dense_2_layer_call_and_return_conditional_losses_35590

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp�0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/kernel/Regularizer/mulMul)Dense_2/kernel/Regularizer/mul/x:output:0*Dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/bias/Regularizer/mulMul'Dense_2/bias/Regularizer/mul/x:output:0(Dense_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp/^Dense_2/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2`
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_35654E
7dense_2_bias_regularizer_l2loss_readvariableop_resource:
identity��.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp�
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp7dense_2_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
:*
dtype0�
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/bias/Regularizer/mulMul'Dense_2/bias/Regularizer/mul/x:output:0(Dense_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ^
IdentityIdentity Dense_2/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^Dense_2/bias/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp
�5
�
G__inference_sequential_1_layer_call_and_return_conditional_losses_35191

inputs 
dense_1_35151:	�
dense_1_35153:
dense_2_35156:
dense_2_35158:
output_35161:
output_35163:
identity��Dense_1/StatefulPartitionedCall�.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp�0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp�Dense_2/StatefulPartitionedCall�.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp�0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp�Output/StatefulPartitionedCall�-Output/bias/Regularizer/L2Loss/ReadVariableOp�/Output/kernel/Regularizer/L2Loss/ReadVariableOp�
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_34998�
Dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_35151dense_1_35153*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_35019�
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_35156dense_2_35158*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_35044�
Output/StatefulPartitionedCallStatefulPartitionedCall(Dense_2/StatefulPartitionedCall:output:0output_35161output_35163*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Output_layer_call_and_return_conditional_losses_35069
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_35151*
_output_shapes
:	�*
dtype0�
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0*Dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_35153*
_output_shapes
:*
dtype0�
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0(Dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ~
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_35156*
_output_shapes

:*
dtype0�
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/kernel/Regularizer/mulMul)Dense_2/kernel/Regularizer/mul/x:output:0*Dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_35158*
_output_shapes
:*
dtype0�
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Dense_2/bias/Regularizer/mulMul'Dense_2/bias/Regularizer/mul/x:output:0(Dense_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_35161*
_output_shapes

:*
dtype0�
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_35163*
_output_shapes
:*
dtype0�
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
Output/bias/Regularizer/mulMul&Output/bias/Regularizer/mul/x:output:0'Output/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^Dense_1/StatefulPartitionedCall/^Dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^Dense_2/StatefulPartitionedCall/^Dense_2/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp^Output/StatefulPartitionedCall.^Output/bias/Regularizer/L2Loss/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : 2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2`
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2`
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2^
-Output/bias/Regularizer/L2Loss/ReadVariableOp-Output/bias/Regularizer/L2Loss/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_Output_layer_call_fn_35599

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Output_layer_call_and_return_conditional_losses_35069o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_34960
x
unknown:	�
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *#
fR
__inference_<lambda>_34941f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:E A
"
_output_shapes
:

_user_specified_namex"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default}
*
x%
serving_default_x:03
output_0'
StatefulPartitionedCall:0tensorflow/serving/predict:�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias"
_tf_keras_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias"
_tf_keras_layer
J
0
1
"2
#3
*4
+5"
trackable_list_wrapper
J
0
1
"2
#3
*4
+5"
trackable_list_wrapper
J
,0
-1
.2
/3
04
15"
trackable_list_wrapper
�
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
�
7trace_0
8trace_1
9trace_2
:trace_32�
,__inference_sequential_1_layer_call_fn_35206
,__inference_sequential_1_layer_call_fn_35267
,__inference_sequential_1_layer_call_fn_35404
,__inference_sequential_1_layer_call_fn_35421�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z7trace_0z8trace_1z9trace_2z:trace_3
�
;trace_0
<trace_1
=trace_2
>trace_32�
G__inference_sequential_1_layer_call_and_return_conditional_losses_35100
G__inference_sequential_1_layer_call_and_return_conditional_losses_35144
G__inference_sequential_1_layer_call_and_return_conditional_losses_35472
G__inference_sequential_1_layer_call_and_return_conditional_losses_35523�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z;trace_0z<trace_1z=trace_2z>trace_3
�B�
 __inference__wrapped_model_34988flatten_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
?
_variables
@_iterations
A_learning_rate
B_index_dict
C
_momentums
D_velocities
E_update_step_xla"
experimentalOptimizer
,
Fserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ltrace_02�
'__inference_flatten_layer_call_fn_35528�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zLtrace_0
�
Mtrace_02�
B__inference_flatten_layer_call_and_return_conditional_losses_35534�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zMtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Strace_02�
'__inference_Dense_1_layer_call_fn_35543�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zStrace_0
�
Ttrace_02�
B__inference_Dense_1_layer_call_and_return_conditional_losses_35562�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zTtrace_0
!:	�2Dense_1/kernel
:2Dense_1/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
Ztrace_02�
'__inference_Dense_2_layer_call_fn_35571�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zZtrace_0
�
[trace_02�
B__inference_Dense_2_layer_call_and_return_conditional_losses_35590�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z[trace_0
 :2Dense_2/kernel
:2Dense_2/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
�
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
atrace_02�
&__inference_Output_layer_call_fn_35599�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zatrace_0
�
btrace_02�
A__inference_Output_layer_call_and_return_conditional_losses_35618�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zbtrace_0
:2Output/kernel
:2Output/bias
�
ctrace_02�
__inference_loss_fn_0_35627�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zctrace_0
�
dtrace_02�
__inference_loss_fn_1_35636�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zdtrace_0
�
etrace_02�
__inference_loss_fn_2_35645�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zetrace_0
�
ftrace_02�
__inference_loss_fn_3_35654�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zftrace_0
�
gtrace_02�
__inference_loss_fn_4_35663�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zgtrace_0
�
htrace_02�
__inference_loss_fn_5_35672�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zhtrace_0
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_sequential_1_layer_call_fn_35206flatten_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_1_layer_call_fn_35267flatten_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_1_layer_call_fn_35404inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_1_layer_call_fn_35421inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_1_layer_call_and_return_conditional_losses_35100flatten_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_1_layer_call_and_return_conditional_losses_35144flatten_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_1_layer_call_and_return_conditional_losses_35472inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_1_layer_call_and_return_conditional_losses_35523inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
~
@0
k1
l2
m3
n4
o5
p6
q7
r8
s9
t10
u11
v12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
J
k0
m1
o2
q3
s4
u5"
trackable_list_wrapper
J
l0
n1
p2
r3
t4
v5"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
#__inference_signature_wrapper_34960x"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_flatten_layer_call_fn_35528inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_flatten_layer_call_and_return_conditional_losses_35534inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_Dense_1_layer_call_fn_35543inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_Dense_1_layer_call_and_return_conditional_losses_35562inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_Dense_2_layer_call_fn_35571inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_Dense_2_layer_call_and_return_conditional_losses_35590inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_Output_layer_call_fn_35599inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_Output_layer_call_and_return_conditional_losses_35618inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_35627"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_35636"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_35645"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_3_35654"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_4_35663"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_5_35672"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
N
w	variables
x	keras_api
	ytotal
	zcount"
_tf_keras_metric
^
{	variables
|	keras_api
	}total
	~count

_fn_kwargs"
_tf_keras_metric
&:$	�2Adam/m/Dense_1/kernel
&:$	�2Adam/v/Dense_1/kernel
:2Adam/m/Dense_1/bias
:2Adam/v/Dense_1/bias
%:#2Adam/m/Dense_2/kernel
%:#2Adam/v/Dense_2/kernel
:2Adam/m/Dense_2/bias
:2Adam/v/Dense_2/bias
$:"2Adam/m/Output/kernel
$:"2Adam/v/Output/kernel
:2Adam/m/Output/bias
:2Adam/v/Output/bias
.
y0
z1"
trackable_list_wrapper
-
w	variables"
_generic_user_object
:  (2total
:  (2count
.
}0
~1"
trackable_list_wrapper
-
{	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
B__inference_Dense_1_layer_call_and_return_conditional_losses_35562d0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
'__inference_Dense_1_layer_call_fn_35543Y0�-
&�#
!�
inputs����������
� "!�
unknown����������
B__inference_Dense_2_layer_call_and_return_conditional_losses_35590c"#/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
'__inference_Dense_2_layer_call_fn_35571X"#/�,
%�"
 �
inputs���������
� "!�
unknown����������
A__inference_Output_layer_call_and_return_conditional_losses_35618c*+/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
&__inference_Output_layer_call_fn_35599X*+/�,
%�"
 �
inputs���������
� "!�
unknown����������
 __inference__wrapped_model_34988u"#*+:�7
0�-
+�(
flatten_input���������
� "/�,
*
Output �
output����������
B__inference_flatten_layer_call_and_return_conditional_losses_35534d3�0
)�&
$�!
inputs���������
� "-�*
#� 
tensor_0����������
� �
'__inference_flatten_layer_call_fn_35528Y3�0
)�&
$�!
inputs���������
� ""�
unknown����������C
__inference_loss_fn_0_35627$�

� 
� "�
unknown C
__inference_loss_fn_1_35636$�

� 
� "�
unknown C
__inference_loss_fn_2_35645$"�

� 
� "�
unknown C
__inference_loss_fn_3_35654$#�

� 
� "�
unknown C
__inference_loss_fn_4_35663$*�

� 
� "�
unknown C
__inference_loss_fn_5_35672$+�

� 
� "�
unknown �
G__inference_sequential_1_layer_call_and_return_conditional_losses_35100z"#*+B�?
8�5
+�(
flatten_input���������
p

 
� ",�)
"�
tensor_0���������
� �
G__inference_sequential_1_layer_call_and_return_conditional_losses_35144z"#*+B�?
8�5
+�(
flatten_input���������
p 

 
� ",�)
"�
tensor_0���������
� �
G__inference_sequential_1_layer_call_and_return_conditional_losses_35472s"#*+;�8
1�.
$�!
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
G__inference_sequential_1_layer_call_and_return_conditional_losses_35523s"#*+;�8
1�.
$�!
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
,__inference_sequential_1_layer_call_fn_35206o"#*+B�?
8�5
+�(
flatten_input���������
p

 
� "!�
unknown����������
,__inference_sequential_1_layer_call_fn_35267o"#*+B�?
8�5
+�(
flatten_input���������
p 

 
� "!�
unknown����������
,__inference_sequential_1_layer_call_fn_35404h"#*+;�8
1�.
$�!
inputs���������
p

 
� "!�
unknown����������
,__inference_sequential_1_layer_call_fn_35421h"#*+;�8
1�.
$�!
inputs���������
p 

 
� "!�
unknown����������
#__inference_signature_wrapper_34960`"#*+*�'
� 
 �

x�
x"*�'
%
output_0�
output_0