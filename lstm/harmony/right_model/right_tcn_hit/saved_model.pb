х„.
Аг
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
†
BatchToSpaceND

input"T
block_shape"Tblock_shape
crops"Tcrops
output"T"	
Ttype" 
Tblock_shapetype0:
2	"
Tcropstype0:
2	
А
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
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resourceИ
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
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
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
©
SpaceToBatchND

input"T
block_shape"Tblock_shape
paddings"	Tpaddings
output"T"	
Ttype" 
Tblock_shapetype0:
2	"
	Tpaddingstype0:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Ѕ
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
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ч
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.12.02unknown8ан(
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
Д
Adam/v/Output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameAdam/v/Output/kernel
}
(Adam/v/Output/kernel/Read/ReadVariableOpReadVariableOpAdam/v/Output/kernel*
_output_shapes

: *
dtype0
Д
Adam/m/Output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameAdam/m/Output/kernel
}
(Adam/m/Output/kernel/Read/ReadVariableOpReadVariableOpAdam/m/Output/kernel*
_output_shapes

: *
dtype0
~
Adam/v/Dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/v/Dense_2/bias
w
'Adam/v/Dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/Dense_2/bias*
_output_shapes
: *
dtype0
~
Adam/m/Dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/m/Dense_2/bias
w
'Adam/m/Dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/Dense_2/bias*
_output_shapes
: *
dtype0
Ж
Adam/v/Dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/v/Dense_2/kernel

)Adam/v/Dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/Dense_2/kernel*
_output_shapes

:  *
dtype0
Ж
Adam/m/Dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/m/Dense_2/kernel

)Adam/m/Dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/Dense_2/kernel*
_output_shapes

:  *
dtype0
~
Adam/v/Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/v/Dense_1/bias
w
'Adam/v/Dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/Dense_1/bias*
_output_shapes
: *
dtype0
~
Adam/m/Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/m/Dense_1/bias
w
'Adam/m/Dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/Dense_1/bias*
_output_shapes
: *
dtype0
Ж
Adam/v/Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/v/Dense_1/kernel

)Adam/v/Dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/Dense_1/kernel*
_output_shapes

:  *
dtype0
Ж
Adam/m/Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/m/Dense_1/kernel

)Adam/m/Dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/Dense_1/kernel*
_output_shapes

:  *
dtype0
Ѓ
+Adam/v/TCN_1/residual_block_3/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/v/TCN_1/residual_block_3/conv1D_1/bias
І
?Adam/v/TCN_1/residual_block_3/conv1D_1/bias/Read/ReadVariableOpReadVariableOp+Adam/v/TCN_1/residual_block_3/conv1D_1/bias*
_output_shapes
: *
dtype0
Ѓ
+Adam/m/TCN_1/residual_block_3/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/m/TCN_1/residual_block_3/conv1D_1/bias
І
?Adam/m/TCN_1/residual_block_3/conv1D_1/bias/Read/ReadVariableOpReadVariableOp+Adam/m/TCN_1/residual_block_3/conv1D_1/bias*
_output_shapes
: *
dtype0
Ї
-Adam/v/TCN_1/residual_block_3/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *>
shared_name/-Adam/v/TCN_1/residual_block_3/conv1D_1/kernel
≥
AAdam/v/TCN_1/residual_block_3/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp-Adam/v/TCN_1/residual_block_3/conv1D_1/kernel*"
_output_shapes
:  *
dtype0
Ї
-Adam/m/TCN_1/residual_block_3/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *>
shared_name/-Adam/m/TCN_1/residual_block_3/conv1D_1/kernel
≥
AAdam/m/TCN_1/residual_block_3/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp-Adam/m/TCN_1/residual_block_3/conv1D_1/kernel*"
_output_shapes
:  *
dtype0
Ѓ
+Adam/v/TCN_1/residual_block_3/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/v/TCN_1/residual_block_3/conv1D_0/bias
І
?Adam/v/TCN_1/residual_block_3/conv1D_0/bias/Read/ReadVariableOpReadVariableOp+Adam/v/TCN_1/residual_block_3/conv1D_0/bias*
_output_shapes
: *
dtype0
Ѓ
+Adam/m/TCN_1/residual_block_3/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/m/TCN_1/residual_block_3/conv1D_0/bias
І
?Adam/m/TCN_1/residual_block_3/conv1D_0/bias/Read/ReadVariableOpReadVariableOp+Adam/m/TCN_1/residual_block_3/conv1D_0/bias*
_output_shapes
: *
dtype0
Ї
-Adam/v/TCN_1/residual_block_3/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *>
shared_name/-Adam/v/TCN_1/residual_block_3/conv1D_0/kernel
≥
AAdam/v/TCN_1/residual_block_3/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp-Adam/v/TCN_1/residual_block_3/conv1D_0/kernel*"
_output_shapes
:  *
dtype0
Ї
-Adam/m/TCN_1/residual_block_3/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *>
shared_name/-Adam/m/TCN_1/residual_block_3/conv1D_0/kernel
≥
AAdam/m/TCN_1/residual_block_3/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp-Adam/m/TCN_1/residual_block_3/conv1D_0/kernel*"
_output_shapes
:  *
dtype0
Ѓ
+Adam/v/TCN_1/residual_block_2/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/v/TCN_1/residual_block_2/conv1D_1/bias
І
?Adam/v/TCN_1/residual_block_2/conv1D_1/bias/Read/ReadVariableOpReadVariableOp+Adam/v/TCN_1/residual_block_2/conv1D_1/bias*
_output_shapes
: *
dtype0
Ѓ
+Adam/m/TCN_1/residual_block_2/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/m/TCN_1/residual_block_2/conv1D_1/bias
І
?Adam/m/TCN_1/residual_block_2/conv1D_1/bias/Read/ReadVariableOpReadVariableOp+Adam/m/TCN_1/residual_block_2/conv1D_1/bias*
_output_shapes
: *
dtype0
Ї
-Adam/v/TCN_1/residual_block_2/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *>
shared_name/-Adam/v/TCN_1/residual_block_2/conv1D_1/kernel
≥
AAdam/v/TCN_1/residual_block_2/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp-Adam/v/TCN_1/residual_block_2/conv1D_1/kernel*"
_output_shapes
:  *
dtype0
Ї
-Adam/m/TCN_1/residual_block_2/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *>
shared_name/-Adam/m/TCN_1/residual_block_2/conv1D_1/kernel
≥
AAdam/m/TCN_1/residual_block_2/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp-Adam/m/TCN_1/residual_block_2/conv1D_1/kernel*"
_output_shapes
:  *
dtype0
Ѓ
+Adam/v/TCN_1/residual_block_2/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/v/TCN_1/residual_block_2/conv1D_0/bias
І
?Adam/v/TCN_1/residual_block_2/conv1D_0/bias/Read/ReadVariableOpReadVariableOp+Adam/v/TCN_1/residual_block_2/conv1D_0/bias*
_output_shapes
: *
dtype0
Ѓ
+Adam/m/TCN_1/residual_block_2/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/m/TCN_1/residual_block_2/conv1D_0/bias
І
?Adam/m/TCN_1/residual_block_2/conv1D_0/bias/Read/ReadVariableOpReadVariableOp+Adam/m/TCN_1/residual_block_2/conv1D_0/bias*
_output_shapes
: *
dtype0
Ї
-Adam/v/TCN_1/residual_block_2/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *>
shared_name/-Adam/v/TCN_1/residual_block_2/conv1D_0/kernel
≥
AAdam/v/TCN_1/residual_block_2/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp-Adam/v/TCN_1/residual_block_2/conv1D_0/kernel*"
_output_shapes
:  *
dtype0
Ї
-Adam/m/TCN_1/residual_block_2/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *>
shared_name/-Adam/m/TCN_1/residual_block_2/conv1D_0/kernel
≥
AAdam/m/TCN_1/residual_block_2/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp-Adam/m/TCN_1/residual_block_2/conv1D_0/kernel*"
_output_shapes
:  *
dtype0
Ѓ
+Adam/v/TCN_1/residual_block_1/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/v/TCN_1/residual_block_1/conv1D_1/bias
І
?Adam/v/TCN_1/residual_block_1/conv1D_1/bias/Read/ReadVariableOpReadVariableOp+Adam/v/TCN_1/residual_block_1/conv1D_1/bias*
_output_shapes
: *
dtype0
Ѓ
+Adam/m/TCN_1/residual_block_1/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/m/TCN_1/residual_block_1/conv1D_1/bias
І
?Adam/m/TCN_1/residual_block_1/conv1D_1/bias/Read/ReadVariableOpReadVariableOp+Adam/m/TCN_1/residual_block_1/conv1D_1/bias*
_output_shapes
: *
dtype0
Ї
-Adam/v/TCN_1/residual_block_1/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *>
shared_name/-Adam/v/TCN_1/residual_block_1/conv1D_1/kernel
≥
AAdam/v/TCN_1/residual_block_1/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp-Adam/v/TCN_1/residual_block_1/conv1D_1/kernel*"
_output_shapes
:  *
dtype0
Ї
-Adam/m/TCN_1/residual_block_1/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *>
shared_name/-Adam/m/TCN_1/residual_block_1/conv1D_1/kernel
≥
AAdam/m/TCN_1/residual_block_1/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp-Adam/m/TCN_1/residual_block_1/conv1D_1/kernel*"
_output_shapes
:  *
dtype0
Ѓ
+Adam/v/TCN_1/residual_block_1/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/v/TCN_1/residual_block_1/conv1D_0/bias
І
?Adam/v/TCN_1/residual_block_1/conv1D_0/bias/Read/ReadVariableOpReadVariableOp+Adam/v/TCN_1/residual_block_1/conv1D_0/bias*
_output_shapes
: *
dtype0
Ѓ
+Adam/m/TCN_1/residual_block_1/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/m/TCN_1/residual_block_1/conv1D_0/bias
І
?Adam/m/TCN_1/residual_block_1/conv1D_0/bias/Read/ReadVariableOpReadVariableOp+Adam/m/TCN_1/residual_block_1/conv1D_0/bias*
_output_shapes
: *
dtype0
Ї
-Adam/v/TCN_1/residual_block_1/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *>
shared_name/-Adam/v/TCN_1/residual_block_1/conv1D_0/kernel
≥
AAdam/v/TCN_1/residual_block_1/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp-Adam/v/TCN_1/residual_block_1/conv1D_0/kernel*"
_output_shapes
:  *
dtype0
Ї
-Adam/m/TCN_1/residual_block_1/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *>
shared_name/-Adam/m/TCN_1/residual_block_1/conv1D_0/kernel
≥
AAdam/m/TCN_1/residual_block_1/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp-Adam/m/TCN_1/residual_block_1/conv1D_0/kernel*"
_output_shapes
:  *
dtype0
Љ
2Adam/v/TCN_1/residual_block_0/matching_conv1D/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42Adam/v/TCN_1/residual_block_0/matching_conv1D/bias
µ
FAdam/v/TCN_1/residual_block_0/matching_conv1D/bias/Read/ReadVariableOpReadVariableOp2Adam/v/TCN_1/residual_block_0/matching_conv1D/bias*
_output_shapes
: *
dtype0
Љ
2Adam/m/TCN_1/residual_block_0/matching_conv1D/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42Adam/m/TCN_1/residual_block_0/matching_conv1D/bias
µ
FAdam/m/TCN_1/residual_block_0/matching_conv1D/bias/Read/ReadVariableOpReadVariableOp2Adam/m/TCN_1/residual_block_0/matching_conv1D/bias*
_output_shapes
: *
dtype0
»
4Adam/v/TCN_1/residual_block_0/matching_conv1D/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *E
shared_name64Adam/v/TCN_1/residual_block_0/matching_conv1D/kernel
Ѕ
HAdam/v/TCN_1/residual_block_0/matching_conv1D/kernel/Read/ReadVariableOpReadVariableOp4Adam/v/TCN_1/residual_block_0/matching_conv1D/kernel*"
_output_shapes
: *
dtype0
»
4Adam/m/TCN_1/residual_block_0/matching_conv1D/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *E
shared_name64Adam/m/TCN_1/residual_block_0/matching_conv1D/kernel
Ѕ
HAdam/m/TCN_1/residual_block_0/matching_conv1D/kernel/Read/ReadVariableOpReadVariableOp4Adam/m/TCN_1/residual_block_0/matching_conv1D/kernel*"
_output_shapes
: *
dtype0
Ѓ
+Adam/v/TCN_1/residual_block_0/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/v/TCN_1/residual_block_0/conv1D_1/bias
І
?Adam/v/TCN_1/residual_block_0/conv1D_1/bias/Read/ReadVariableOpReadVariableOp+Adam/v/TCN_1/residual_block_0/conv1D_1/bias*
_output_shapes
: *
dtype0
Ѓ
+Adam/m/TCN_1/residual_block_0/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/m/TCN_1/residual_block_0/conv1D_1/bias
І
?Adam/m/TCN_1/residual_block_0/conv1D_1/bias/Read/ReadVariableOpReadVariableOp+Adam/m/TCN_1/residual_block_0/conv1D_1/bias*
_output_shapes
: *
dtype0
Ї
-Adam/v/TCN_1/residual_block_0/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *>
shared_name/-Adam/v/TCN_1/residual_block_0/conv1D_1/kernel
≥
AAdam/v/TCN_1/residual_block_0/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp-Adam/v/TCN_1/residual_block_0/conv1D_1/kernel*"
_output_shapes
:  *
dtype0
Ї
-Adam/m/TCN_1/residual_block_0/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *>
shared_name/-Adam/m/TCN_1/residual_block_0/conv1D_1/kernel
≥
AAdam/m/TCN_1/residual_block_0/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp-Adam/m/TCN_1/residual_block_0/conv1D_1/kernel*"
_output_shapes
:  *
dtype0
Ѓ
+Adam/v/TCN_1/residual_block_0/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/v/TCN_1/residual_block_0/conv1D_0/bias
І
?Adam/v/TCN_1/residual_block_0/conv1D_0/bias/Read/ReadVariableOpReadVariableOp+Adam/v/TCN_1/residual_block_0/conv1D_0/bias*
_output_shapes
: *
dtype0
Ѓ
+Adam/m/TCN_1/residual_block_0/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/m/TCN_1/residual_block_0/conv1D_0/bias
І
?Adam/m/TCN_1/residual_block_0/conv1D_0/bias/Read/ReadVariableOpReadVariableOp+Adam/m/TCN_1/residual_block_0/conv1D_0/bias*
_output_shapes
: *
dtype0
Ї
-Adam/v/TCN_1/residual_block_0/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/v/TCN_1/residual_block_0/conv1D_0/kernel
≥
AAdam/v/TCN_1/residual_block_0/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp-Adam/v/TCN_1/residual_block_0/conv1D_0/kernel*"
_output_shapes
: *
dtype0
Ї
-Adam/m/TCN_1/residual_block_0/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/m/TCN_1/residual_block_0/conv1D_0/kernel
≥
AAdam/m/TCN_1/residual_block_0/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp-Adam/m/TCN_1/residual_block_0/conv1D_0/kernel*"
_output_shapes
: *
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
†
$TCN_1/residual_block_3/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$TCN_1/residual_block_3/conv1D_1/bias
Щ
8TCN_1/residual_block_3/conv1D_1/bias/Read/ReadVariableOpReadVariableOp$TCN_1/residual_block_3/conv1D_1/bias*
_output_shapes
: *
dtype0
ђ
&TCN_1/residual_block_3/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *7
shared_name(&TCN_1/residual_block_3/conv1D_1/kernel
•
:TCN_1/residual_block_3/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp&TCN_1/residual_block_3/conv1D_1/kernel*"
_output_shapes
:  *
dtype0
†
$TCN_1/residual_block_3/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$TCN_1/residual_block_3/conv1D_0/bias
Щ
8TCN_1/residual_block_3/conv1D_0/bias/Read/ReadVariableOpReadVariableOp$TCN_1/residual_block_3/conv1D_0/bias*
_output_shapes
: *
dtype0
ђ
&TCN_1/residual_block_3/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *7
shared_name(&TCN_1/residual_block_3/conv1D_0/kernel
•
:TCN_1/residual_block_3/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp&TCN_1/residual_block_3/conv1D_0/kernel*"
_output_shapes
:  *
dtype0
†
$TCN_1/residual_block_2/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$TCN_1/residual_block_2/conv1D_1/bias
Щ
8TCN_1/residual_block_2/conv1D_1/bias/Read/ReadVariableOpReadVariableOp$TCN_1/residual_block_2/conv1D_1/bias*
_output_shapes
: *
dtype0
ђ
&TCN_1/residual_block_2/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *7
shared_name(&TCN_1/residual_block_2/conv1D_1/kernel
•
:TCN_1/residual_block_2/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp&TCN_1/residual_block_2/conv1D_1/kernel*"
_output_shapes
:  *
dtype0
†
$TCN_1/residual_block_2/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$TCN_1/residual_block_2/conv1D_0/bias
Щ
8TCN_1/residual_block_2/conv1D_0/bias/Read/ReadVariableOpReadVariableOp$TCN_1/residual_block_2/conv1D_0/bias*
_output_shapes
: *
dtype0
ђ
&TCN_1/residual_block_2/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *7
shared_name(&TCN_1/residual_block_2/conv1D_0/kernel
•
:TCN_1/residual_block_2/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp&TCN_1/residual_block_2/conv1D_0/kernel*"
_output_shapes
:  *
dtype0
†
$TCN_1/residual_block_1/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$TCN_1/residual_block_1/conv1D_1/bias
Щ
8TCN_1/residual_block_1/conv1D_1/bias/Read/ReadVariableOpReadVariableOp$TCN_1/residual_block_1/conv1D_1/bias*
_output_shapes
: *
dtype0
ђ
&TCN_1/residual_block_1/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *7
shared_name(&TCN_1/residual_block_1/conv1D_1/kernel
•
:TCN_1/residual_block_1/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp&TCN_1/residual_block_1/conv1D_1/kernel*"
_output_shapes
:  *
dtype0
†
$TCN_1/residual_block_1/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$TCN_1/residual_block_1/conv1D_0/bias
Щ
8TCN_1/residual_block_1/conv1D_0/bias/Read/ReadVariableOpReadVariableOp$TCN_1/residual_block_1/conv1D_0/bias*
_output_shapes
: *
dtype0
ђ
&TCN_1/residual_block_1/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *7
shared_name(&TCN_1/residual_block_1/conv1D_0/kernel
•
:TCN_1/residual_block_1/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp&TCN_1/residual_block_1/conv1D_0/kernel*"
_output_shapes
:  *
dtype0
Ѓ
+TCN_1/residual_block_0/matching_conv1D/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+TCN_1/residual_block_0/matching_conv1D/bias
І
?TCN_1/residual_block_0/matching_conv1D/bias/Read/ReadVariableOpReadVariableOp+TCN_1/residual_block_0/matching_conv1D/bias*
_output_shapes
: *
dtype0
Ї
-TCN_1/residual_block_0/matching_conv1D/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-TCN_1/residual_block_0/matching_conv1D/kernel
≥
ATCN_1/residual_block_0/matching_conv1D/kernel/Read/ReadVariableOpReadVariableOp-TCN_1/residual_block_0/matching_conv1D/kernel*"
_output_shapes
: *
dtype0
†
$TCN_1/residual_block_0/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$TCN_1/residual_block_0/conv1D_1/bias
Щ
8TCN_1/residual_block_0/conv1D_1/bias/Read/ReadVariableOpReadVariableOp$TCN_1/residual_block_0/conv1D_1/bias*
_output_shapes
: *
dtype0
ђ
&TCN_1/residual_block_0/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *7
shared_name(&TCN_1/residual_block_0/conv1D_1/kernel
•
:TCN_1/residual_block_0/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp&TCN_1/residual_block_0/conv1D_1/kernel*"
_output_shapes
:  *
dtype0
†
$TCN_1/residual_block_0/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$TCN_1/residual_block_0/conv1D_0/bias
Щ
8TCN_1/residual_block_0/conv1D_0/bias/Read/ReadVariableOpReadVariableOp$TCN_1/residual_block_0/conv1D_0/bias*
_output_shapes
: *
dtype0
ђ
&TCN_1/residual_block_0/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&TCN_1/residual_block_0/conv1D_0/kernel
•
:TCN_1/residual_block_0/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp&TCN_1/residual_block_0/conv1D_0/kernel*"
_output_shapes
: *
dtype0
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
: *
shared_nameOutput/kernel
o
!Output/kernel/Read/ReadVariableOpReadVariableOpOutput/kernel*
_output_shapes

: *
dtype0
p
Dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameDense_2/bias
i
 Dense_2/bias/Read/ReadVariableOpReadVariableOpDense_2/bias*
_output_shapes
: *
dtype0
x
Dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_nameDense_2/kernel
q
"Dense_2/kernel/Read/ReadVariableOpReadVariableOpDense_2/kernel*
_output_shapes

:  *
dtype0
p
Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameDense_1/bias
i
 Dense_1/bias/Read/ReadVariableOpReadVariableOpDense_1/bias*
_output_shapes
: *
dtype0
x
Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_nameDense_1/kernel
q
"Dense_1/kernel/Read/ReadVariableOpReadVariableOpDense_1/kernel*
_output_shapes

:  *
dtype0
l
serving_default_xPlaceholder*#
_output_shapes
:В*
dtype0*
shape:В
ц
StatefulPartitionedCallStatefulPartitionedCallserving_default_x&TCN_1/residual_block_0/conv1D_0/kernel$TCN_1/residual_block_0/conv1D_0/bias&TCN_1/residual_block_0/conv1D_1/kernel$TCN_1/residual_block_0/conv1D_1/bias-TCN_1/residual_block_0/matching_conv1D/kernel+TCN_1/residual_block_0/matching_conv1D/bias&TCN_1/residual_block_1/conv1D_0/kernel$TCN_1/residual_block_1/conv1D_0/bias&TCN_1/residual_block_1/conv1D_1/kernel$TCN_1/residual_block_1/conv1D_1/bias&TCN_1/residual_block_2/conv1D_0/kernel$TCN_1/residual_block_2/conv1D_0/bias&TCN_1/residual_block_2/conv1D_1/kernel$TCN_1/residual_block_2/conv1D_1/bias&TCN_1/residual_block_3/conv1D_0/kernel$TCN_1/residual_block_3/conv1D_0/bias&TCN_1/residual_block_3/conv1D_1/kernel$TCN_1/residual_block_3/conv1D_1/biasDense_1/kernelDense_1/biasDense_2/kernelDense_2/biasOutput/kernelOutput/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_12673

NoOpNoOp
СҐ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ћ°
valueј°BЉ° Bі°
и
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
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
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	dilations
skip_connections
residual_blocks
layers_outputs
residual_block_0
residual_block_1
residual_block_2
residual_block_3
slicer_layer*
¶
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*
¶
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias*
¶
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias*
Ї
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17
#18
$19
+20
,21
322
423*
Ї
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17
#18
$19
+20
,21
322
423*
,
G0
H1
I2
J3
K4
L5* 
∞
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
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
Rtrace_0
Strace_1
Ttrace_2
Utrace_3* 
6
Vtrace_0
Wtrace_1
Xtrace_2
Ytrace_3* 
* 
Б
Z
_variables
[_iterations
\_learning_rate
]_index_dict
^
_momentums
__velocities
`_update_step_xla*

aserving_default* 
К
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17*
К
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17*
* 
У
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

gtrace_0
htrace_1* 

itrace_0
jtrace_1* 
* 
* 
 
0
1
2
3*
* 
е
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

qlayers
rshape_match_conv
sfinal_activation
tconv1D_0
uAct_Conv1D_0
v
SDropout_0
wconv1D_1
xAct_Conv1D_1
y
SDropout_1
zAct_Conv_Blocks
rmatching_conv1D
sAct_Res_Block*
ф
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+А&call_and_return_all_conditional_losses
Бlayers
Вshape_match_conv
Гfinal_activation
Дconv1D_0
ЕAct_Conv1D_0
Ж
SDropout_0
Зconv1D_1
ИAct_Conv1D_1
Й
SDropout_1
КAct_Conv_Blocks
Вmatching_identity
ГAct_Res_Block*
щ
Л	variables
Мtrainable_variables
Нregularization_losses
О	keras_api
П__call__
+Р&call_and_return_all_conditional_losses
Сlayers
Тshape_match_conv
Уfinal_activation
Фconv1D_0
ХAct_Conv1D_0
Ц
SDropout_0
Чconv1D_1
ШAct_Conv1D_1
Щ
SDropout_1
ЪAct_Conv_Blocks
Тmatching_identity
УAct_Res_Block*
щ
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+†&call_and_return_all_conditional_losses
°layers
Ґshape_match_conv
£final_activation
§conv1D_0
•Act_Conv1D_0
¶
SDropout_0
Іconv1D_1
®Act_Conv1D_1
©
SDropout_1
™Act_Conv_Blocks
Ґmatching_identity
£Act_Res_Block*
Ф
Ђ	variables
ђtrainable_variables
≠regularization_losses
Ѓ	keras_api
ѓ__call__
+∞&call_and_return_all_conditional_losses* 

#0
$1*

#0
$1*

G0
H1* 
Ш
±non_trainable_variables
≤layers
≥metrics
 іlayer_regularization_losses
µlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

ґtrace_0* 

Јtrace_0* 
^X
VARIABLE_VALUEDense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEDense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

+0
,1*

+0
,1*

I0
J1* 
Ш
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Љlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

љtrace_0* 

Њtrace_0* 
^X
VARIABLE_VALUEDense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEDense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

30
41*

30
41*

K0
L1* 
Ш
њnon_trainable_variables
јlayers
Ѕmetrics
 ¬layer_regularization_losses
√layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

ƒtrace_0* 

≈trace_0* 
]W
VARIABLE_VALUEOutput/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEOutput/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&TCN_1/residual_block_0/conv1D_0/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$TCN_1/residual_block_0/conv1D_0/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&TCN_1/residual_block_0/conv1D_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$TCN_1/residual_block_0/conv1D_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE-TCN_1/residual_block_0/matching_conv1D/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE+TCN_1/residual_block_0/matching_conv1D/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&TCN_1/residual_block_1/conv1D_0/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$TCN_1/residual_block_1/conv1D_0/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&TCN_1/residual_block_1/conv1D_1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$TCN_1/residual_block_1/conv1D_1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&TCN_1/residual_block_2/conv1D_0/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$TCN_1/residual_block_2/conv1D_0/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&TCN_1/residual_block_2/conv1D_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$TCN_1/residual_block_2/conv1D_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&TCN_1/residual_block_3/conv1D_0/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$TCN_1/residual_block_3/conv1D_0/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&TCN_1/residual_block_3/conv1D_1/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$TCN_1/residual_block_3/conv1D_1/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*

∆trace_0* 

«trace_0* 

»trace_0* 

…trace_0* 

 trace_0* 

Ћtrace_0* 
* 
 
0
1
2
3*

ћ0
Ќ1*
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
≤
[0
ќ1
ѕ2
–3
—4
“5
”6
‘7
’8
÷9
„10
Ў11
ў12
Џ13
џ14
№15
Ё16
ё17
я18
а19
б20
в21
г22
д23
е24
ж25
з26
и27
й28
к29
л30
м31
н32
о33
п34
р35
с36
т37
у38
ф39
х40
ц41
ч42
ш43
щ44
ъ45
ы46
ь47
э48*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
“
ќ0
–1
“2
‘3
÷4
Ў5
Џ6
№7
ё8
а9
в10
д11
ж12
и13
к14
м15
о16
р17
т18
ф19
ц20
ш21
ъ22
ь23*
“
ѕ0
—1
”2
’3
„4
ў5
џ6
Ё7
я8
б9
г10
е11
з12
й13
л14
н15
п16
с17
у18
х19
ч20
щ21
ы22
э23*
* 
* 
* 
'
0
1
2
3
4*
* 
* 
* 
* 
* 
* 
* 
.
50
61
72
83
94
:5*
.
50
61
72
83
94
:5*
* 
Ш
юnon_trainable_variables
€layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*
* 
* 
5
t0
u1
v2
w3
x4
y5
z6*
ѕ
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
З__call__
+И&call_and_return_all_conditional_losses

9kernel
:bias
!Й_jit_compiled_convolution_op*
Ф
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses* 
ѕ
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses

5kernel
6bias
!Ц_jit_compiled_convolution_op*
Ф
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses* 
ђ
Э	variables
Юtrainable_variables
Яregularization_losses
†	keras_api
°__call__
+Ґ&call_and_return_all_conditional_losses
£_random_generator* 
ѕ
§	variables
•trainable_variables
¶regularization_losses
І	keras_api
®__call__
+©&call_and_return_all_conditional_losses

7kernel
8bias
!™_jit_compiled_convolution_op*
Ф
Ђ	variables
ђtrainable_variables
≠regularization_losses
Ѓ	keras_api
ѓ__call__
+∞&call_and_return_all_conditional_losses* 
ђ
±	variables
≤trainable_variables
≥regularization_losses
і	keras_api
µ__call__
+ґ&call_and_return_all_conditional_losses
Ј_random_generator* 
Ф
Є	variables
єtrainable_variables
Їregularization_losses
ї	keras_api
Љ__call__
+љ&call_and_return_all_conditional_losses* 
 
;0
<1
=2
>3*
 
;0
<1
=2
>3*
* 
Ъ
Њnon_trainable_variables
њlayers
јmetrics
 Ѕlayer_regularization_losses
¬layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses*
* 
* 
<
Д0
Е1
Ж2
З3
И4
Й5
К6*
Ф
√	variables
ƒtrainable_variables
≈regularization_losses
∆	keras_api
«__call__
+»&call_and_return_all_conditional_losses* 
Ф
…	variables
 trainable_variables
Ћregularization_losses
ћ	keras_api
Ќ__call__
+ќ&call_and_return_all_conditional_losses* 
ѕ
ѕ	variables
–trainable_variables
—regularization_losses
“	keras_api
”__call__
+‘&call_and_return_all_conditional_losses

;kernel
<bias
!’_jit_compiled_convolution_op*
Ф
÷	variables
„trainable_variables
Ўregularization_losses
ў	keras_api
Џ__call__
+џ&call_and_return_all_conditional_losses* 
ђ
№	variables
Ёtrainable_variables
ёregularization_losses
я	keras_api
а__call__
+б&call_and_return_all_conditional_losses
в_random_generator* 
ѕ
г	variables
дtrainable_variables
еregularization_losses
ж	keras_api
з__call__
+и&call_and_return_all_conditional_losses

=kernel
>bias
!й_jit_compiled_convolution_op*
Ф
к	variables
лtrainable_variables
мregularization_losses
н	keras_api
о__call__
+п&call_and_return_all_conditional_losses* 
ђ
р	variables
сtrainable_variables
тregularization_losses
у	keras_api
ф__call__
+х&call_and_return_all_conditional_losses
ц_random_generator* 
Ф
ч	variables
шtrainable_variables
щregularization_losses
ъ	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses* 
 
?0
@1
A2
B3*
 
?0
@1
A2
B3*
* 
Ю
эnon_trainable_variables
юlayers
€metrics
 Аlayer_regularization_losses
Бlayer_metrics
Л	variables
Мtrainable_variables
Нregularization_losses
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses*
* 
* 
<
Ф0
Х1
Ц2
Ч3
Ш4
Щ5
Ъ6*
Ф
В	variables
Гtrainable_variables
Дregularization_losses
Е	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses* 
Ф
И	variables
Йtrainable_variables
Кregularization_losses
Л	keras_api
М__call__
+Н&call_and_return_all_conditional_losses* 
ѕ
О	variables
Пtrainable_variables
Рregularization_losses
С	keras_api
Т__call__
+У&call_and_return_all_conditional_losses

?kernel
@bias
!Ф_jit_compiled_convolution_op*
Ф
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses* 
ђ
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+†&call_and_return_all_conditional_losses
°_random_generator* 
ѕ
Ґ	variables
£trainable_variables
§regularization_losses
•	keras_api
¶__call__
+І&call_and_return_all_conditional_losses

Akernel
Bbias
!®_jit_compiled_convolution_op*
Ф
©	variables
™trainable_variables
Ђregularization_losses
ђ	keras_api
≠__call__
+Ѓ&call_and_return_all_conditional_losses* 
ђ
ѓ	variables
∞trainable_variables
±regularization_losses
≤	keras_api
≥__call__
+і&call_and_return_all_conditional_losses
µ_random_generator* 
Ф
ґ	variables
Јtrainable_variables
Єregularization_losses
є	keras_api
Ї__call__
+ї&call_and_return_all_conditional_losses* 
 
C0
D1
E2
F3*
 
C0
D1
E2
F3*
* 
Ю
Љnon_trainable_variables
љlayers
Њmetrics
 њlayer_regularization_losses
јlayer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses*
* 
* 
<
§0
•1
¶2
І3
®4
©5
™6*
Ф
Ѕ	variables
¬trainable_variables
√regularization_losses
ƒ	keras_api
≈__call__
+∆&call_and_return_all_conditional_losses* 
Ф
«	variables
»trainable_variables
…regularization_losses
 	keras_api
Ћ__call__
+ћ&call_and_return_all_conditional_losses* 
ѕ
Ќ	variables
ќtrainable_variables
ѕregularization_losses
–	keras_api
—__call__
+“&call_and_return_all_conditional_losses

Ckernel
Dbias
!”_jit_compiled_convolution_op*
Ф
‘	variables
’trainable_variables
÷regularization_losses
„	keras_api
Ў__call__
+ў&call_and_return_all_conditional_losses* 
ђ
Џ	variables
џtrainable_variables
№regularization_losses
Ё	keras_api
ё__call__
+я&call_and_return_all_conditional_losses
а_random_generator* 
ѕ
б	variables
вtrainable_variables
гregularization_losses
д	keras_api
е__call__
+ж&call_and_return_all_conditional_losses

Ekernel
Fbias
!з_jit_compiled_convolution_op*
Ф
и	variables
йtrainable_variables
кregularization_losses
л	keras_api
м__call__
+н&call_and_return_all_conditional_losses* 
ђ
о	variables
пtrainable_variables
рregularization_losses
с	keras_api
т__call__
+у&call_and_return_all_conditional_losses
ф_random_generator* 
Ф
х	variables
цtrainable_variables
чregularization_losses
ш	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses* 
* 
* 
* 
Ь
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
€layer_metrics
Ђ	variables
ђtrainable_variables
≠regularization_losses
ѓ__call__
+∞&call_and_return_all_conditional_losses
'∞"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

G0
H1* 
* 
* 
* 
* 
* 
* 

I0
J1* 
* 
* 
* 
* 
* 
* 

K0
L1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
А	variables
Б	keras_api

Вtotal

Гcount*
M
Д	variables
Е	keras_api

Жtotal

Зcount
И
_fn_kwargs*
xr
VARIABLE_VALUE-Adam/m/TCN_1/residual_block_0/conv1D_0/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE-Adam/v/TCN_1/residual_block_0/conv1D_0/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE+Adam/m/TCN_1/residual_block_0/conv1D_0/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE+Adam/v/TCN_1/residual_block_0/conv1D_0/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE-Adam/m/TCN_1/residual_block_0/conv1D_1/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE-Adam/v/TCN_1/residual_block_0/conv1D_1/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE+Adam/m/TCN_1/residual_block_0/conv1D_1/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE+Adam/v/TCN_1/residual_block_0/conv1D_1/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE4Adam/m/TCN_1/residual_block_0/matching_conv1D/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUE4Adam/v/TCN_1/residual_block_0/matching_conv1D/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE2Adam/m/TCN_1/residual_block_0/matching_conv1D/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE2Adam/v/TCN_1/residual_block_0/matching_conv1D/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/m/TCN_1/residual_block_1/conv1D_0/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/v/TCN_1/residual_block_1/conv1D_0/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+Adam/m/TCN_1/residual_block_1/conv1D_0/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+Adam/v/TCN_1/residual_block_1/conv1D_0/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/m/TCN_1/residual_block_1/conv1D_1/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/v/TCN_1/residual_block_1/conv1D_1/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+Adam/m/TCN_1/residual_block_1/conv1D_1/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+Adam/v/TCN_1/residual_block_1/conv1D_1/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/m/TCN_1/residual_block_2/conv1D_0/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/v/TCN_1/residual_block_2/conv1D_0/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+Adam/m/TCN_1/residual_block_2/conv1D_0/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+Adam/v/TCN_1/residual_block_2/conv1D_0/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/m/TCN_1/residual_block_2/conv1D_1/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/v/TCN_1/residual_block_2/conv1D_1/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+Adam/m/TCN_1/residual_block_2/conv1D_1/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+Adam/v/TCN_1/residual_block_2/conv1D_1/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/m/TCN_1/residual_block_3/conv1D_0/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/v/TCN_1/residual_block_3/conv1D_0/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+Adam/m/TCN_1/residual_block_3/conv1D_0/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+Adam/v/TCN_1/residual_block_3/conv1D_0/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/m/TCN_1/residual_block_3/conv1D_1/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/v/TCN_1/residual_block_3/conv1D_1/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+Adam/m/TCN_1/residual_block_3/conv1D_1/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+Adam/v/TCN_1/residual_block_3/conv1D_1/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/Dense_1/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/Dense_1/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/Dense_1/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/Dense_1/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/Dense_2/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/Dense_2/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/Dense_2/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/Dense_2/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/Output/kernel2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/Output/kernel2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/Output/bias2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/Output/bias2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
* 
C
t0
u1
v2
w3
x4
y5
z6
r7
s8*
* 
* 
* 

90
:1*

90
:1*
* 
Ю
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ь
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses* 
* 
* 

50
61*

50
61*
* 
Ю
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ь
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
Эnon_trainable_variables
Юlayers
Яmetrics
 †layer_regularization_losses
°layer_metrics
Э	variables
Юtrainable_variables
Яregularization_losses
°__call__
+Ґ&call_and_return_all_conditional_losses
'Ґ"call_and_return_conditional_losses* 

Ґtrace_0
£trace_1* 

§trace_0
•trace_1* 
* 

70
81*

70
81*
* 
Ю
¶non_trainable_variables
Іlayers
®metrics
 ©layer_regularization_losses
™layer_metrics
§	variables
•trainable_variables
¶regularization_losses
®__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ь
Ђnon_trainable_variables
ђlayers
≠metrics
 Ѓlayer_regularization_losses
ѓlayer_metrics
Ђ	variables
ђtrainable_variables
≠regularization_losses
ѓ__call__
+∞&call_and_return_all_conditional_losses
'∞"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
∞non_trainable_variables
±layers
≤metrics
 ≥layer_regularization_losses
іlayer_metrics
±	variables
≤trainable_variables
≥regularization_losses
µ__call__
+ґ&call_and_return_all_conditional_losses
'ґ"call_and_return_conditional_losses* 

µtrace_0
ґtrace_1* 

Јtrace_0
Єtrace_1* 
* 
* 
* 
* 
Ь
єnon_trainable_variables
Їlayers
їmetrics
 Љlayer_regularization_losses
љlayer_metrics
Є	variables
єtrainable_variables
Їregularization_losses
Љ__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses* 
* 
* 
* 
L
Д0
Е1
Ж2
З3
И4
Й5
К6
В7
Г8*
* 
* 
* 
* 
* 
* 
Ь
Њnon_trainable_variables
њlayers
јmetrics
 Ѕlayer_regularization_losses
¬layer_metrics
√	variables
ƒtrainable_variables
≈regularization_losses
«__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
√non_trainable_variables
ƒlayers
≈metrics
 ∆layer_regularization_losses
«layer_metrics
…	variables
 trainable_variables
Ћregularization_losses
Ќ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses* 
* 
* 

;0
<1*

;0
<1*
* 
Ю
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
ѕ	variables
–trainable_variables
—regularization_losses
”__call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ь
Ќnon_trainable_variables
ќlayers
ѕmetrics
 –layer_regularization_losses
—layer_metrics
÷	variables
„trainable_variables
Ўregularization_losses
Џ__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
№	variables
Ёtrainable_variables
ёregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses* 

„trace_0
Ўtrace_1* 

ўtrace_0
Џtrace_1* 
* 

=0
>1*

=0
>1*
* 
Ю
џnon_trainable_variables
№layers
Ёmetrics
 ёlayer_regularization_losses
яlayer_metrics
г	variables
дtrainable_variables
еregularization_losses
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ь
аnon_trainable_variables
бlayers
вmetrics
 гlayer_regularization_losses
дlayer_metrics
к	variables
лtrainable_variables
мregularization_losses
о__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
р	variables
сtrainable_variables
тregularization_losses
ф__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses* 

кtrace_0
лtrace_1* 

мtrace_0
нtrace_1* 
* 
* 
* 
* 
Ь
оnon_trainable_variables
пlayers
рmetrics
 сlayer_regularization_losses
тlayer_metrics
ч	variables
шtrainable_variables
щregularization_losses
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses* 
* 
* 
* 
L
Ф0
Х1
Ц2
Ч3
Ш4
Щ5
Ъ6
Т7
У8*
* 
* 
* 
* 
* 
* 
Ь
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
В	variables
Гtrainable_variables
Дregularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
И	variables
Йtrainable_variables
Кregularization_losses
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses* 
* 
* 

?0
@1*

?0
@1*
* 
Ю
эnon_trainable_variables
юlayers
€metrics
 Аlayer_regularization_losses
Бlayer_metrics
О	variables
Пtrainable_variables
Рregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ь
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses* 

Мtrace_0
Нtrace_1* 

Оtrace_0
Пtrace_1* 
* 

A0
B1*

A0
B1*
* 
Ю
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
Ґ	variables
£trainable_variables
§regularization_losses
¶__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ь
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
©	variables
™trainable_variables
Ђregularization_losses
≠__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
ѓ	variables
∞trainable_variables
±regularization_losses
≥__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses* 

Яtrace_0
†trace_1* 

°trace_0
Ґtrace_1* 
* 
* 
* 
* 
Ь
£non_trainable_variables
§layers
•metrics
 ¶layer_regularization_losses
Іlayer_metrics
ґ	variables
Јtrainable_variables
Єregularization_losses
Ї__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses* 
* 
* 
* 
L
§0
•1
¶2
І3
®4
©5
™6
Ґ7
£8*
* 
* 
* 
* 
* 
* 
Ь
®non_trainable_variables
©layers
™metrics
 Ђlayer_regularization_losses
ђlayer_metrics
Ѕ	variables
¬trainable_variables
√regularization_losses
≈__call__
+∆&call_and_return_all_conditional_losses
'∆"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
≠non_trainable_variables
Ѓlayers
ѓmetrics
 ∞layer_regularization_losses
±layer_metrics
«	variables
»trainable_variables
…regularization_losses
Ћ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses* 
* 
* 

C0
D1*

C0
D1*
* 
Ю
≤non_trainable_variables
≥layers
іmetrics
 µlayer_regularization_losses
ґlayer_metrics
Ќ	variables
ќtrainable_variables
ѕregularization_losses
—__call__
+“&call_and_return_all_conditional_losses
'“"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ь
Јnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
‘	variables
’trainable_variables
÷regularization_losses
Ў__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
Љnon_trainable_variables
љlayers
Њmetrics
 њlayer_regularization_losses
јlayer_metrics
Џ	variables
џtrainable_variables
№regularization_losses
ё__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses* 

Ѕtrace_0
¬trace_1* 

√trace_0
ƒtrace_1* 
* 

E0
F1*

E0
F1*
* 
Ю
≈non_trainable_variables
∆layers
«metrics
 »layer_regularization_losses
…layer_metrics
б	variables
вtrainable_variables
гregularization_losses
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ь
 non_trainable_variables
Ћlayers
ћmetrics
 Ќlayer_regularization_losses
ќlayer_metrics
и	variables
йtrainable_variables
кregularization_losses
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
ѕnon_trainable_variables
–layers
—metrics
 “layer_regularization_losses
”layer_metrics
о	variables
пtrainable_variables
рregularization_losses
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses* 

‘trace_0
’trace_1* 

÷trace_0
„trace_1* 
* 
* 
* 
* 
Ь
Ўnon_trainable_variables
ўlayers
Џmetrics
 џlayer_regularization_losses
№layer_metrics
х	variables
цtrainable_variables
чregularization_losses
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 

В0
Г1*

А	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ж0
З1*

Д	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
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
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ё
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameDense_1/kernelDense_1/biasDense_2/kernelDense_2/biasOutput/kernelOutput/bias&TCN_1/residual_block_0/conv1D_0/kernel$TCN_1/residual_block_0/conv1D_0/bias&TCN_1/residual_block_0/conv1D_1/kernel$TCN_1/residual_block_0/conv1D_1/bias-TCN_1/residual_block_0/matching_conv1D/kernel+TCN_1/residual_block_0/matching_conv1D/bias&TCN_1/residual_block_1/conv1D_0/kernel$TCN_1/residual_block_1/conv1D_0/bias&TCN_1/residual_block_1/conv1D_1/kernel$TCN_1/residual_block_1/conv1D_1/bias&TCN_1/residual_block_2/conv1D_0/kernel$TCN_1/residual_block_2/conv1D_0/bias&TCN_1/residual_block_2/conv1D_1/kernel$TCN_1/residual_block_2/conv1D_1/bias&TCN_1/residual_block_3/conv1D_0/kernel$TCN_1/residual_block_3/conv1D_0/bias&TCN_1/residual_block_3/conv1D_1/kernel$TCN_1/residual_block_3/conv1D_1/bias	iterationlearning_rate-Adam/m/TCN_1/residual_block_0/conv1D_0/kernel-Adam/v/TCN_1/residual_block_0/conv1D_0/kernel+Adam/m/TCN_1/residual_block_0/conv1D_0/bias+Adam/v/TCN_1/residual_block_0/conv1D_0/bias-Adam/m/TCN_1/residual_block_0/conv1D_1/kernel-Adam/v/TCN_1/residual_block_0/conv1D_1/kernel+Adam/m/TCN_1/residual_block_0/conv1D_1/bias+Adam/v/TCN_1/residual_block_0/conv1D_1/bias4Adam/m/TCN_1/residual_block_0/matching_conv1D/kernel4Adam/v/TCN_1/residual_block_0/matching_conv1D/kernel2Adam/m/TCN_1/residual_block_0/matching_conv1D/bias2Adam/v/TCN_1/residual_block_0/matching_conv1D/bias-Adam/m/TCN_1/residual_block_1/conv1D_0/kernel-Adam/v/TCN_1/residual_block_1/conv1D_0/kernel+Adam/m/TCN_1/residual_block_1/conv1D_0/bias+Adam/v/TCN_1/residual_block_1/conv1D_0/bias-Adam/m/TCN_1/residual_block_1/conv1D_1/kernel-Adam/v/TCN_1/residual_block_1/conv1D_1/kernel+Adam/m/TCN_1/residual_block_1/conv1D_1/bias+Adam/v/TCN_1/residual_block_1/conv1D_1/bias-Adam/m/TCN_1/residual_block_2/conv1D_0/kernel-Adam/v/TCN_1/residual_block_2/conv1D_0/kernel+Adam/m/TCN_1/residual_block_2/conv1D_0/bias+Adam/v/TCN_1/residual_block_2/conv1D_0/bias-Adam/m/TCN_1/residual_block_2/conv1D_1/kernel-Adam/v/TCN_1/residual_block_2/conv1D_1/kernel+Adam/m/TCN_1/residual_block_2/conv1D_1/bias+Adam/v/TCN_1/residual_block_2/conv1D_1/bias-Adam/m/TCN_1/residual_block_3/conv1D_0/kernel-Adam/v/TCN_1/residual_block_3/conv1D_0/kernel+Adam/m/TCN_1/residual_block_3/conv1D_0/bias+Adam/v/TCN_1/residual_block_3/conv1D_0/bias-Adam/m/TCN_1/residual_block_3/conv1D_1/kernel-Adam/v/TCN_1/residual_block_3/conv1D_1/kernel+Adam/m/TCN_1/residual_block_3/conv1D_1/bias+Adam/v/TCN_1/residual_block_3/conv1D_1/biasAdam/m/Dense_1/kernelAdam/v/Dense_1/kernelAdam/m/Dense_1/biasAdam/v/Dense_1/biasAdam/m/Dense_2/kernelAdam/v/Dense_2/kernelAdam/m/Dense_2/biasAdam/v/Dense_2/biasAdam/m/Output/kernelAdam/v/Output/kernelAdam/m/Output/biasAdam/v/Output/biastotal_1count_1totalcountConst*[
TinT
R2P*
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
GPU 2J 8В *'
f"R 
__inference__traced_save_16236
ў
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameDense_1/kernelDense_1/biasDense_2/kernelDense_2/biasOutput/kernelOutput/bias&TCN_1/residual_block_0/conv1D_0/kernel$TCN_1/residual_block_0/conv1D_0/bias&TCN_1/residual_block_0/conv1D_1/kernel$TCN_1/residual_block_0/conv1D_1/bias-TCN_1/residual_block_0/matching_conv1D/kernel+TCN_1/residual_block_0/matching_conv1D/bias&TCN_1/residual_block_1/conv1D_0/kernel$TCN_1/residual_block_1/conv1D_0/bias&TCN_1/residual_block_1/conv1D_1/kernel$TCN_1/residual_block_1/conv1D_1/bias&TCN_1/residual_block_2/conv1D_0/kernel$TCN_1/residual_block_2/conv1D_0/bias&TCN_1/residual_block_2/conv1D_1/kernel$TCN_1/residual_block_2/conv1D_1/bias&TCN_1/residual_block_3/conv1D_0/kernel$TCN_1/residual_block_3/conv1D_0/bias&TCN_1/residual_block_3/conv1D_1/kernel$TCN_1/residual_block_3/conv1D_1/bias	iterationlearning_rate-Adam/m/TCN_1/residual_block_0/conv1D_0/kernel-Adam/v/TCN_1/residual_block_0/conv1D_0/kernel+Adam/m/TCN_1/residual_block_0/conv1D_0/bias+Adam/v/TCN_1/residual_block_0/conv1D_0/bias-Adam/m/TCN_1/residual_block_0/conv1D_1/kernel-Adam/v/TCN_1/residual_block_0/conv1D_1/kernel+Adam/m/TCN_1/residual_block_0/conv1D_1/bias+Adam/v/TCN_1/residual_block_0/conv1D_1/bias4Adam/m/TCN_1/residual_block_0/matching_conv1D/kernel4Adam/v/TCN_1/residual_block_0/matching_conv1D/kernel2Adam/m/TCN_1/residual_block_0/matching_conv1D/bias2Adam/v/TCN_1/residual_block_0/matching_conv1D/bias-Adam/m/TCN_1/residual_block_1/conv1D_0/kernel-Adam/v/TCN_1/residual_block_1/conv1D_0/kernel+Adam/m/TCN_1/residual_block_1/conv1D_0/bias+Adam/v/TCN_1/residual_block_1/conv1D_0/bias-Adam/m/TCN_1/residual_block_1/conv1D_1/kernel-Adam/v/TCN_1/residual_block_1/conv1D_1/kernel+Adam/m/TCN_1/residual_block_1/conv1D_1/bias+Adam/v/TCN_1/residual_block_1/conv1D_1/bias-Adam/m/TCN_1/residual_block_2/conv1D_0/kernel-Adam/v/TCN_1/residual_block_2/conv1D_0/kernel+Adam/m/TCN_1/residual_block_2/conv1D_0/bias+Adam/v/TCN_1/residual_block_2/conv1D_0/bias-Adam/m/TCN_1/residual_block_2/conv1D_1/kernel-Adam/v/TCN_1/residual_block_2/conv1D_1/kernel+Adam/m/TCN_1/residual_block_2/conv1D_1/bias+Adam/v/TCN_1/residual_block_2/conv1D_1/bias-Adam/m/TCN_1/residual_block_3/conv1D_0/kernel-Adam/v/TCN_1/residual_block_3/conv1D_0/kernel+Adam/m/TCN_1/residual_block_3/conv1D_0/bias+Adam/v/TCN_1/residual_block_3/conv1D_0/bias-Adam/m/TCN_1/residual_block_3/conv1D_1/kernel-Adam/v/TCN_1/residual_block_3/conv1D_1/kernel+Adam/m/TCN_1/residual_block_3/conv1D_1/bias+Adam/v/TCN_1/residual_block_3/conv1D_1/biasAdam/m/Dense_1/kernelAdam/v/Dense_1/kernelAdam/m/Dense_1/biasAdam/v/Dense_1/biasAdam/m/Dense_2/kernelAdam/v/Dense_2/kernelAdam/m/Dense_2/biasAdam/v/Dense_2/biasAdam/m/Output/kernelAdam/v/Output/kernelAdam/m/Output/biasAdam/v/Output/biastotal_1count_1totalcount*Z
TinS
Q2O*
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
GPU 2J 8В **
f%R#
!__inference__traced_restore_16480еќ$
я
§
__inference_loss_fn_5_15585D
6output_bias_regularizer_l2loss_readvariableop_resource:
identityИҐ-Output/bias/Regularizer/L2Loss/ReadVariableOp†
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp6output_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
:*
dtype0А
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ф
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
∞
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15625

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15740

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Њ
Ф
'__inference_Dense_2_layer_call_fn_15484

inputs
unknown:  
	unknown_0: 
identityИҐStatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_13397o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
й=
А

G__inference_sequential_1_layer_call_and_return_conditional_losses_13453
tcn_1_input!
tcn_1_13316: 
tcn_1_13318: !
tcn_1_13320:  
tcn_1_13322: !
tcn_1_13324: 
tcn_1_13326: !
tcn_1_13328:  
tcn_1_13330: !
tcn_1_13332:  
tcn_1_13334: !
tcn_1_13336:  
tcn_1_13338: !
tcn_1_13340:  
tcn_1_13342: !
tcn_1_13344:  
tcn_1_13346: !
tcn_1_13348:  
tcn_1_13350: 
dense_1_13373:  
dense_1_13375: 
dense_2_13398:  
dense_2_13400: 
output_13423: 
output_13425:
identityИҐDense_1/StatefulPartitionedCallҐ.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpҐDense_2/StatefulPartitionedCallҐ.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpҐ0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpҐOutput/StatefulPartitionedCallҐ-Output/bias/Regularizer/L2Loss/ReadVariableOpҐ/Output/kernel/Regularizer/L2Loss/ReadVariableOpҐTCN_1/StatefulPartitionedCall÷
TCN_1/StatefulPartitionedCallStatefulPartitionedCalltcn_1_inputtcn_1_13316tcn_1_13318tcn_1_13320tcn_1_13322tcn_1_13324tcn_1_13326tcn_1_13328tcn_1_13330tcn_1_13332tcn_1_13334tcn_1_13336tcn_1_13338tcn_1_13340tcn_1_13342tcn_1_13344tcn_1_13346tcn_1_13348tcn_1_13350*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_TCN_1_layer_call_and_return_conditional_losses_13315Й
Dense_1/StatefulPartitionedCallStatefulPartitionedCall&TCN_1/StatefulPartitionedCall:output:0dense_1_13373dense_1_13375*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_13372Л
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_13398dense_2_13400*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_13397З
Output/StatefulPartitionedCallStatefulPartitionedCall(Dense_2/StatefulPartitionedCall:output:0output_13423output_13425*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_Output_layer_call_and_return_conditional_losses_13422~
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_13373*
_output_shapes

:  *
dtype0Ж
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0*Dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_13375*
_output_shapes
: *
dtype0В
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0(Dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ~
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_13398*
_output_shapes

:  *
dtype0Ж
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
Dense_2/kernel/Regularizer/mulMul)Dense_2/kernel/Regularizer/mul/x:output:0*Dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_13400*
_output_shapes
: *
dtype0В
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
Dense_2/bias/Regularizer/mulMul'Dense_2/bias/Regularizer/mul/x:output:0(Dense_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_13423*
_output_shapes

: *
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ъ
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_13425*
_output_shapes
:*
dtype0А
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ф
Output/bias/Regularizer/mulMul&Output/bias/Regularizer/mul/x:output:0'Output/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€х
NoOpNoOp ^Dense_1/StatefulPartitionedCall/^Dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^Dense_2/StatefulPartitionedCall/^Dense_2/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp^Output/StatefulPartitionedCall.^Output/bias/Regularizer/L2Loss/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp^TCN_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€В: : : : : : : : : : : : : : : : : : : : : : : : 2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2`
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2`
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2^
-Output/bias/Regularizer/L2Loss/ReadVariableOp-Output/bias/Regularizer/L2Loss/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp2>
TCN_1/StatefulPartitionedCallTCN_1/StatefulPartitionedCall:Y U
,
_output_shapes
:€€€€€€€€€В
%
_user_specified_nameTCN_1_input
Ў•
В
__inference_<lambda>_12618
xn
Xsequential_1_tcn_1_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource: Z
Lsequential_1_tcn_1_residual_block_0_conv1d_0_biasadd_readvariableop_resource: n
Xsequential_1_tcn_1_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  Z
Lsequential_1_tcn_1_residual_block_0_conv1d_1_biasadd_readvariableop_resource: u
_sequential_1_tcn_1_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource: a
Ssequential_1_tcn_1_residual_block_0_matching_conv1d_biasadd_readvariableop_resource: n
Xsequential_1_tcn_1_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  Z
Lsequential_1_tcn_1_residual_block_1_conv1d_0_biasadd_readvariableop_resource: n
Xsequential_1_tcn_1_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  Z
Lsequential_1_tcn_1_residual_block_1_conv1d_1_biasadd_readvariableop_resource: n
Xsequential_1_tcn_1_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  Z
Lsequential_1_tcn_1_residual_block_2_conv1d_0_biasadd_readvariableop_resource: n
Xsequential_1_tcn_1_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  Z
Lsequential_1_tcn_1_residual_block_2_conv1d_1_biasadd_readvariableop_resource: n
Xsequential_1_tcn_1_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  Z
Lsequential_1_tcn_1_residual_block_3_conv1d_0_biasadd_readvariableop_resource: n
Xsequential_1_tcn_1_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  Z
Lsequential_1_tcn_1_residual_block_3_conv1d_1_biasadd_readvariableop_resource: E
3sequential_1_dense_1_matmul_readvariableop_resource:  B
4sequential_1_dense_1_biasadd_readvariableop_resource: E
3sequential_1_dense_2_matmul_readvariableop_resource:  B
4sequential_1_dense_2_biasadd_readvariableop_resource: D
2sequential_1_output_matmul_readvariableop_resource: A
3sequential_1_output_biasadd_readvariableop_resource:
identityИҐ+sequential_1/Dense_1/BiasAdd/ReadVariableOpҐ*sequential_1/Dense_1/MatMul/ReadVariableOpҐ+sequential_1/Dense_2/BiasAdd/ReadVariableOpҐ*sequential_1/Dense_2/MatMul/ReadVariableOpҐ*sequential_1/Output/BiasAdd/ReadVariableOpҐ)sequential_1/Output/MatMul/ReadVariableOpҐCsequential_1/TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpҐOsequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐCsequential_1/TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpҐOsequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐJsequential_1/TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpҐVsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpҐCsequential_1/TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpҐOsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐCsequential_1/TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpҐOsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐCsequential_1/TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpҐOsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐCsequential_1/TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpҐOsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐCsequential_1/TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpҐOsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐCsequential_1/TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpҐOsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ
9sequential_1/TCN_1/residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ђ
0sequential_1/TCN_1/residual_block_0/conv1D_0/PadPadxBsequential_1/TCN_1/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*#
_output_shapes
:ДН
Bsequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Ж
>sequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims9sequential_1/TCN_1/residual_block_0/conv1D_0/Pad:output:0Ksequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*'
_output_shapes
:Дм
Osequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpXsequential_1_tcn_1_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Ж
Dsequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : І
@sequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsWsequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Msequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ђ
3sequential_1/TCN_1/residual_block_0/conv1D_0/Conv1DConv2DGsequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims:output:0Isequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*'
_output_shapes
:В *
paddingVALID*
strides
“
;sequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze<sequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D:output:0*
T0*#
_output_shapes
:В *
squeeze_dims

э€€€€€€€€ћ
Csequential_1/TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpLsequential_1_tcn_1_residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0А
4sequential_1/TCN_1/residual_block_0/conv1D_0/BiasAddBiasAddDsequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/Squeeze:output:0Ksequential_1/TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:В ™
5sequential_1/TCN_1/residual_block_0/Act_Conv1D_0/ReluRelu=sequential_1/TCN_1/residual_block_0/conv1D_0/BiasAdd:output:0*
T0*#
_output_shapes
:В ґ
7sequential_1/TCN_1/residual_block_0/SDropout_0/IdentityIdentityCsequential_1/TCN_1/residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*#
_output_shapes
:В Ґ
9sequential_1/TCN_1/residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       л
0sequential_1/TCN_1/residual_block_0/conv1D_1/PadPad@sequential_1/TCN_1/residual_block_0/SDropout_0/Identity:output:0Bsequential_1/TCN_1/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*#
_output_shapes
:Д Н
Bsequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Ж
>sequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims9sequential_1/TCN_1/residual_block_0/conv1D_1/Pad:output:0Ksequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*'
_output_shapes
:Д м
Osequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpXsequential_1_tcn_1_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Ж
Dsequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : І
@sequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsWsequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Msequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ђ
3sequential_1/TCN_1/residual_block_0/conv1D_1/Conv1DConv2DGsequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims:output:0Isequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*'
_output_shapes
:В *
paddingVALID*
strides
“
;sequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze<sequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D:output:0*
T0*#
_output_shapes
:В *
squeeze_dims

э€€€€€€€€ћ
Csequential_1/TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpLsequential_1_tcn_1_residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0А
4sequential_1/TCN_1/residual_block_0/conv1D_1/BiasAddBiasAddDsequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/Squeeze:output:0Ksequential_1/TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:В ™
5sequential_1/TCN_1/residual_block_0/Act_Conv1D_1/ReluRelu=sequential_1/TCN_1/residual_block_0/conv1D_1/BiasAdd:output:0*
T0*#
_output_shapes
:В ґ
7sequential_1/TCN_1/residual_block_0/SDropout_1/IdentityIdentityCsequential_1/TCN_1/residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*#
_output_shapes
:В ∞
8sequential_1/TCN_1/residual_block_0/Act_Conv_Blocks/ReluRelu@sequential_1/TCN_1/residual_block_0/SDropout_1/Identity:output:0*
T0*#
_output_shapes
:В Ф
Isequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€№
Esequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsxRsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*'
_output_shapes
:Въ
Vsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp_sequential_1_tcn_1_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Н
Ksequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Љ
Gsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDims^sequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Tsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ј
:sequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1DConv2DNsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0Psequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*'
_output_shapes
:В *
paddingSAME*
strides
а
Bsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueezeCsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D:output:0*
T0*#
_output_shapes
:В *
squeeze_dims

э€€€€€€€€Џ
Jsequential_1/TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOpSsequential_1_tcn_1_residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Х
;sequential_1/TCN_1/residual_block_0/matching_conv1D/BiasAddBiasAddKsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0Rsequential_1/TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:В ф
/sequential_1/TCN_1/residual_block_0/Add_Res/addAddV2Dsequential_1/TCN_1/residual_block_0/matching_conv1D/BiasAdd:output:0Fsequential_1/TCN_1/residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*#
_output_shapes
:В °
6sequential_1/TCN_1/residual_block_0/Act_Res_Block/ReluRelu3sequential_1/TCN_1/residual_block_0/Add_Res/add:z:0*
T0*#
_output_shapes
:В Ґ
9sequential_1/TCN_1/residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       п
0sequential_1/TCN_1/residual_block_1/conv1D_0/PadPadDsequential_1/TCN_1/residual_block_0/Act_Res_Block/Relu:activations:0Bsequential_1/TCN_1/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*#
_output_shapes
:Ж Л
Asequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ђ
`sequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Жї
bsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ґ
]sequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ≥
Zsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        Ш
Nsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:§
Ksequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        л
Bsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND9sequential_1/TCN_1/residual_block_1/conv1D_0/Pad:output:0Wsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Tsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*"
_output_shapes
:C Н
Bsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Ч
>sequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDimsKsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:0Ksequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*&
_output_shapes
:C м
Osequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpXsequential_1_tcn_1_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Ж
Dsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : І
@sequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsWsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Msequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Ђ
3sequential_1/TCN_1/residual_block_1/conv1D_0/Conv1DConv2DGsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims:output:0Isequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*&
_output_shapes
:A *
paddingVALID*
strides
—
;sequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze<sequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D:output:0*
T0*"
_output_shapes
:A *
squeeze_dims

э€€€€€€€€Ш
Nsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:°
Hsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ф
Bsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceNDDsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Wsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Qsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*#
_output_shapes
:В ћ
Csequential_1/TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpLsequential_1_tcn_1_residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0З
4sequential_1/TCN_1/residual_block_1/conv1D_0/BiasAddBiasAddKsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:0Ksequential_1/TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:В ™
5sequential_1/TCN_1/residual_block_1/Act_Conv1D_0/ReluRelu=sequential_1/TCN_1/residual_block_1/conv1D_0/BiasAdd:output:0*
T0*#
_output_shapes
:В ґ
7sequential_1/TCN_1/residual_block_1/SDropout_0/IdentityIdentityCsequential_1/TCN_1/residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*#
_output_shapes
:В Ґ
9sequential_1/TCN_1/residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       л
0sequential_1/TCN_1/residual_block_1/conv1D_1/PadPad@sequential_1/TCN_1/residual_block_1/SDropout_0/Identity:output:0Bsequential_1/TCN_1/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*#
_output_shapes
:Ж Л
Asequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ђ
`sequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Жї
bsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ґ
]sequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ≥
Zsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        Ш
Nsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:§
Ksequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        л
Bsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND9sequential_1/TCN_1/residual_block_1/conv1D_1/Pad:output:0Wsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Tsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*"
_output_shapes
:C Н
Bsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Ч
>sequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDimsKsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:0Ksequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*&
_output_shapes
:C м
Osequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpXsequential_1_tcn_1_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Ж
Dsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : І
@sequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsWsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Msequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Ђ
3sequential_1/TCN_1/residual_block_1/conv1D_1/Conv1DConv2DGsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims:output:0Isequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*&
_output_shapes
:A *
paddingVALID*
strides
—
;sequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze<sequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D:output:0*
T0*"
_output_shapes
:A *
squeeze_dims

э€€€€€€€€Ш
Nsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:°
Hsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ф
Bsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceNDDsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Wsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Qsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*#
_output_shapes
:В ћ
Csequential_1/TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpLsequential_1_tcn_1_residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0З
4sequential_1/TCN_1/residual_block_1/conv1D_1/BiasAddBiasAddKsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:0Ksequential_1/TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:В ™
5sequential_1/TCN_1/residual_block_1/Act_Conv1D_1/ReluRelu=sequential_1/TCN_1/residual_block_1/conv1D_1/BiasAdd:output:0*
T0*#
_output_shapes
:В ґ
7sequential_1/TCN_1/residual_block_1/SDropout_1/IdentityIdentityCsequential_1/TCN_1/residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*#
_output_shapes
:В ∞
8sequential_1/TCN_1/residual_block_1/Act_Conv_Blocks/ReluRelu@sequential_1/TCN_1/residual_block_1/SDropout_1/Identity:output:0*
T0*#
_output_shapes
:В ф
/sequential_1/TCN_1/residual_block_1/Add_Res/addAddV2Dsequential_1/TCN_1/residual_block_0/Act_Res_Block/Relu:activations:0Fsequential_1/TCN_1/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*#
_output_shapes
:В °
6sequential_1/TCN_1/residual_block_1/Act_Res_Block/ReluRelu3sequential_1/TCN_1/residual_block_1/Add_Res/add:z:0*
T0*#
_output_shapes
:В Ґ
9sequential_1/TCN_1/residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       п
0sequential_1/TCN_1/residual_block_2/conv1D_0/PadPadDsequential_1/TCN_1/residual_block_1/Act_Res_Block/Relu:activations:0Bsequential_1/TCN_1/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*#
_output_shapes
:К Л
Asequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ђ
`sequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Кї
bsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ґ
]sequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ≥
Zsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Ш
Nsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:§
Ksequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       л
Bsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND9sequential_1/TCN_1/residual_block_2/conv1D_0/Pad:output:0Wsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Tsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*"
_output_shapes
:# Н
Bsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Ч
>sequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDimsKsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:0Ksequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*&
_output_shapes
:# м
Osequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpXsequential_1_tcn_1_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Ж
Dsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : І
@sequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsWsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Msequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Ђ
3sequential_1/TCN_1/residual_block_2/conv1D_0/Conv1DConv2DGsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims:output:0Isequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*&
_output_shapes
:! *
paddingVALID*
strides
—
;sequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze<sequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D:output:0*
T0*"
_output_shapes
:! *
squeeze_dims

э€€€€€€€€Ш
Nsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:°
Hsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ф
Bsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceNDDsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Wsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Qsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*#
_output_shapes
:В ћ
Csequential_1/TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpLsequential_1_tcn_1_residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0З
4sequential_1/TCN_1/residual_block_2/conv1D_0/BiasAddBiasAddKsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:0Ksequential_1/TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:В ™
5sequential_1/TCN_1/residual_block_2/Act_Conv1D_0/ReluRelu=sequential_1/TCN_1/residual_block_2/conv1D_0/BiasAdd:output:0*
T0*#
_output_shapes
:В ґ
7sequential_1/TCN_1/residual_block_2/SDropout_0/IdentityIdentityCsequential_1/TCN_1/residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*#
_output_shapes
:В Ґ
9sequential_1/TCN_1/residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       л
0sequential_1/TCN_1/residual_block_2/conv1D_1/PadPad@sequential_1/TCN_1/residual_block_2/SDropout_0/Identity:output:0Bsequential_1/TCN_1/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*#
_output_shapes
:К Л
Asequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ђ
`sequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Кї
bsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ґ
]sequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ≥
Zsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Ш
Nsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:§
Ksequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       л
Bsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND9sequential_1/TCN_1/residual_block_2/conv1D_1/Pad:output:0Wsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Tsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*"
_output_shapes
:# Н
Bsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Ч
>sequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDimsKsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:0Ksequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*&
_output_shapes
:# м
Osequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpXsequential_1_tcn_1_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Ж
Dsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : І
@sequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsWsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Msequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Ђ
3sequential_1/TCN_1/residual_block_2/conv1D_1/Conv1DConv2DGsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims:output:0Isequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*&
_output_shapes
:! *
paddingVALID*
strides
—
;sequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze<sequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D:output:0*
T0*"
_output_shapes
:! *
squeeze_dims

э€€€€€€€€Ш
Nsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:°
Hsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ф
Bsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceNDDsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Wsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Qsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*#
_output_shapes
:В ћ
Csequential_1/TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpLsequential_1_tcn_1_residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0З
4sequential_1/TCN_1/residual_block_2/conv1D_1/BiasAddBiasAddKsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:0Ksequential_1/TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:В ™
5sequential_1/TCN_1/residual_block_2/Act_Conv1D_1/ReluRelu=sequential_1/TCN_1/residual_block_2/conv1D_1/BiasAdd:output:0*
T0*#
_output_shapes
:В ґ
7sequential_1/TCN_1/residual_block_2/SDropout_1/IdentityIdentityCsequential_1/TCN_1/residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*#
_output_shapes
:В ∞
8sequential_1/TCN_1/residual_block_2/Act_Conv_Blocks/ReluRelu@sequential_1/TCN_1/residual_block_2/SDropout_1/Identity:output:0*
T0*#
_output_shapes
:В ф
/sequential_1/TCN_1/residual_block_2/Add_Res/addAddV2Dsequential_1/TCN_1/residual_block_1/Act_Res_Block/Relu:activations:0Fsequential_1/TCN_1/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*#
_output_shapes
:В °
6sequential_1/TCN_1/residual_block_2/Act_Res_Block/ReluRelu3sequential_1/TCN_1/residual_block_2/Add_Res/add:z:0*
T0*#
_output_shapes
:В Ґ
9sequential_1/TCN_1/residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       п
0sequential_1/TCN_1/residual_block_3/conv1D_0/PadPadDsequential_1/TCN_1/residual_block_2/Act_Res_Block/Relu:activations:0Bsequential_1/TCN_1/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*#
_output_shapes
:Т Л
Asequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ђ
`sequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Тї
bsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ґ
]sequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ≥
Zsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Ш
Nsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:§
Ksequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       л
Bsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND9sequential_1/TCN_1/residual_block_3/conv1D_0/Pad:output:0Wsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Tsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*"
_output_shapes
: Н
Bsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Ч
>sequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDimsKsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:0Ksequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*&
_output_shapes
: м
Osequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpXsequential_1_tcn_1_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Ж
Dsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : І
@sequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsWsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Msequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Ђ
3sequential_1/TCN_1/residual_block_3/conv1D_0/Conv1DConv2DGsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims:output:0Isequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
—
;sequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze<sequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D:output:0*
T0*"
_output_shapes
: *
squeeze_dims

э€€€€€€€€Ш
Nsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:°
Hsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ф
Bsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceNDDsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Wsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Qsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*#
_output_shapes
:В ћ
Csequential_1/TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpLsequential_1_tcn_1_residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0З
4sequential_1/TCN_1/residual_block_3/conv1D_0/BiasAddBiasAddKsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:0Ksequential_1/TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:В ™
5sequential_1/TCN_1/residual_block_3/Act_Conv1D_0/ReluRelu=sequential_1/TCN_1/residual_block_3/conv1D_0/BiasAdd:output:0*
T0*#
_output_shapes
:В ґ
7sequential_1/TCN_1/residual_block_3/SDropout_0/IdentityIdentityCsequential_1/TCN_1/residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*#
_output_shapes
:В Ґ
9sequential_1/TCN_1/residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       л
0sequential_1/TCN_1/residual_block_3/conv1D_1/PadPad@sequential_1/TCN_1/residual_block_3/SDropout_0/Identity:output:0Bsequential_1/TCN_1/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*#
_output_shapes
:Т Л
Asequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ђ
`sequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Тї
bsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ґ
]sequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ≥
Zsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Ш
Nsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:§
Ksequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       л
Bsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND9sequential_1/TCN_1/residual_block_3/conv1D_1/Pad:output:0Wsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Tsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*"
_output_shapes
: Н
Bsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Ч
>sequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDimsKsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:0Ksequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*&
_output_shapes
: м
Osequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpXsequential_1_tcn_1_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Ж
Dsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : І
@sequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsWsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Msequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Ђ
3sequential_1/TCN_1/residual_block_3/conv1D_1/Conv1DConv2DGsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims:output:0Isequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
—
;sequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze<sequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D:output:0*
T0*"
_output_shapes
: *
squeeze_dims

э€€€€€€€€Ш
Nsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:°
Hsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ф
Bsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceNDDsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Wsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Qsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*#
_output_shapes
:В ћ
Csequential_1/TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpLsequential_1_tcn_1_residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0З
4sequential_1/TCN_1/residual_block_3/conv1D_1/BiasAddBiasAddKsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:0Ksequential_1/TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*#
_output_shapes
:В ™
5sequential_1/TCN_1/residual_block_3/Act_Conv1D_1/ReluRelu=sequential_1/TCN_1/residual_block_3/conv1D_1/BiasAdd:output:0*
T0*#
_output_shapes
:В ґ
7sequential_1/TCN_1/residual_block_3/SDropout_1/IdentityIdentityCsequential_1/TCN_1/residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*#
_output_shapes
:В ∞
8sequential_1/TCN_1/residual_block_3/Act_Conv_Blocks/ReluRelu@sequential_1/TCN_1/residual_block_3/SDropout_1/Identity:output:0*
T0*#
_output_shapes
:В ф
/sequential_1/TCN_1/residual_block_3/Add_Res/addAddV2Dsequential_1/TCN_1/residual_block_2/Act_Res_Block/Relu:activations:0Fsequential_1/TCN_1/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*#
_output_shapes
:В °
6sequential_1/TCN_1/residual_block_3/Act_Res_Block/ReluRelu3sequential_1/TCN_1/residual_block_3/Add_Res/add:z:0*
T0*#
_output_shapes
:В т
+sequential_1/TCN_1/Add_Skip_Connections/addAddV2Fsequential_1/TCN_1/residual_block_0/Act_Conv_Blocks/Relu:activations:0Fsequential_1/TCN_1/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*#
_output_shapes
:В Ё
-sequential_1/TCN_1/Add_Skip_Connections/add_1AddV2/sequential_1/TCN_1/Add_Skip_Connections/add:z:0Fsequential_1/TCN_1/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*#
_output_shapes
:В я
-sequential_1/TCN_1/Add_Skip_Connections/add_2AddV21sequential_1/TCN_1/Add_Skip_Connections/add_1:z:0Fsequential_1/TCN_1/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*#
_output_shapes
:В И
3sequential_1/TCN_1/Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    €€€€    К
5sequential_1/TCN_1/Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            К
5sequential_1/TCN_1/Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ю
-sequential_1/TCN_1/Slice_Output/strided_sliceStridedSlice1sequential_1/TCN_1/Add_Skip_Connections/add_2:z:0<sequential_1/TCN_1/Slice_Output/strided_slice/stack:output:0>sequential_1/TCN_1/Slice_Output/strided_slice/stack_1:output:0>sequential_1/TCN_1/Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask*
shrink_axis_maskЮ
*sequential_1/Dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ї
sequential_1/Dense_1/MatMulMatMul6sequential_1/TCN_1/Slice_Output/strided_slice:output:02sequential_1/Dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: Ь
+sequential_1/Dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ђ
sequential_1/Dense_1/BiasAddBiasAdd%sequential_1/Dense_1/MatMul:product:03sequential_1/Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: q
sequential_1/Dense_1/ReluRelu%sequential_1/Dense_1/BiasAdd:output:0*
T0*
_output_shapes

: Ю
*sequential_1/Dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ђ
sequential_1/Dense_2/MatMulMatMul'sequential_1/Dense_1/Relu:activations:02sequential_1/Dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: Ь
+sequential_1/Dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ђ
sequential_1/Dense_2/BiasAddBiasAdd%sequential_1/Dense_2/MatMul:product:03sequential_1/Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: q
sequential_1/Dense_2/ReluRelu%sequential_1/Dense_2/BiasAdd:output:0*
T0*
_output_shapes

: Ь
)sequential_1/Output/MatMul/ReadVariableOpReadVariableOp2sequential_1_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype0©
sequential_1/Output/MatMulMatMul'sequential_1/Dense_2/Relu:activations:01sequential_1/Output/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:Ъ
*sequential_1/Output/BiasAdd/ReadVariableOpReadVariableOp3sequential_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
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

:ї
NoOpNoOp,^sequential_1/Dense_1/BiasAdd/ReadVariableOp+^sequential_1/Dense_1/MatMul/ReadVariableOp,^sequential_1/Dense_2/BiasAdd/ReadVariableOp+^sequential_1/Dense_2/MatMul/ReadVariableOp+^sequential_1/Output/BiasAdd/ReadVariableOp*^sequential_1/Output/MatMul/ReadVariableOpD^sequential_1/TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpP^sequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpD^sequential_1/TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpP^sequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpK^sequential_1/TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpW^sequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpD^sequential_1/TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpP^sequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpD^sequential_1/TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpP^sequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpD^sequential_1/TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpP^sequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpD^sequential_1/TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpP^sequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpD^sequential_1/TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpP^sequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpD^sequential_1/TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpP^sequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:В: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+sequential_1/Dense_1/BiasAdd/ReadVariableOp+sequential_1/Dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/Dense_1/MatMul/ReadVariableOp*sequential_1/Dense_1/MatMul/ReadVariableOp2Z
+sequential_1/Dense_2/BiasAdd/ReadVariableOp+sequential_1/Dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/Dense_2/MatMul/ReadVariableOp*sequential_1/Dense_2/MatMul/ReadVariableOp2X
*sequential_1/Output/BiasAdd/ReadVariableOp*sequential_1/Output/BiasAdd/ReadVariableOp2V
)sequential_1/Output/MatMul/ReadVariableOp)sequential_1/Output/MatMul/ReadVariableOp2К
Csequential_1/TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpCsequential_1/TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2Ґ
Osequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpOsequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2К
Csequential_1/TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpCsequential_1/TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2Ґ
Osequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpOsequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2Ш
Jsequential_1/TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpJsequential_1/TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2∞
Vsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpVsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2К
Csequential_1/TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpCsequential_1/TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2Ґ
Osequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpOsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2К
Csequential_1/TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpCsequential_1/TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2Ґ
Osequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpOsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2К
Csequential_1/TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpCsequential_1/TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2Ґ
Osequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpOsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2К
Csequential_1/TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpCsequential_1/TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2Ґ
Osequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpOsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2К
Csequential_1/TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpCsequential_1/TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2Ґ
Osequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpOsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2К
Csequential_1/TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpCsequential_1/TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2Ґ
Osequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpOsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:F B
#
_output_shapes
:В

_user_specified_namex
™Ш
о
G__inference_sequential_1_layer_call_and_return_conditional_losses_14660

inputsa
Ktcn_1_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource: M
?tcn_1_residual_block_0_conv1d_0_biasadd_readvariableop_resource: a
Ktcn_1_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  M
?tcn_1_residual_block_0_conv1d_1_biasadd_readvariableop_resource: h
Rtcn_1_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource: T
Ftcn_1_residual_block_0_matching_conv1d_biasadd_readvariableop_resource: a
Ktcn_1_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  M
?tcn_1_residual_block_1_conv1d_0_biasadd_readvariableop_resource: a
Ktcn_1_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  M
?tcn_1_residual_block_1_conv1d_1_biasadd_readvariableop_resource: a
Ktcn_1_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  M
?tcn_1_residual_block_2_conv1d_0_biasadd_readvariableop_resource: a
Ktcn_1_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  M
?tcn_1_residual_block_2_conv1d_1_biasadd_readvariableop_resource: a
Ktcn_1_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  M
?tcn_1_residual_block_3_conv1d_0_biasadd_readvariableop_resource: a
Ktcn_1_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  M
?tcn_1_residual_block_3_conv1d_1_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:  5
'dense_1_biasadd_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource:  5
'dense_2_biasadd_readvariableop_resource: 7
%output_matmul_readvariableop_resource: 4
&output_biasadd_readvariableop_resource:
identityИҐDense_1/BiasAdd/ReadVariableOpҐDense_1/MatMul/ReadVariableOpҐ.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpҐDense_2/BiasAdd/ReadVariableOpҐDense_2/MatMul/ReadVariableOpҐ.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpҐ0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpҐOutput/BiasAdd/ReadVariableOpҐOutput/MatMul/ReadVariableOpҐ-Output/bias/Regularizer/L2Loss/ReadVariableOpҐ/Output/kernel/Regularizer/L2Loss/ReadVariableOpҐ6TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpҐBTCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ6TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpҐBTCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ=TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpҐITCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpҐ6TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpҐBTCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ6TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpҐBTCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ6TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpҐBTCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ6TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpҐBTCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ6TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpҐBTCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ6TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpҐBTCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpХ
,TCN_1/residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       †
#TCN_1/residual_block_0/conv1D_0/PadPadinputs5TCN_1/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€ДА
5TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€и
1TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims,TCN_1/residual_block_0/conv1D_0/Pad:output:0>TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Д“
BTCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpKtcn_1_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0y
7TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
3TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsJTCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0@TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: О
&TCN_1/residual_block_0/conv1D_0/Conv1DConv2D:TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims:output:0<TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingVALID*
strides
Ѕ
.TCN_1/residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze/TCN_1/residual_block_0/conv1D_0/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€≤
6TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp?tcn_1_residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0в
'TCN_1/residual_block_0/conv1D_0/BiasAddBiasAdd7TCN_1/residual_block_0/conv1D_0/Conv1D/Squeeze:output:0>TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
(TCN_1/residual_block_0/Act_Conv1D_0/ReluRelu0TCN_1/residual_block_0/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В •
*TCN_1/residual_block_0/SDropout_0/IdentityIdentity6TCN_1/residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Х
,TCN_1/residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ќ
#TCN_1/residual_block_0/conv1D_1/PadPad3TCN_1/residual_block_0/SDropout_0/Identity:output:05TCN_1/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Д А
5TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€и
1TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims,TCN_1/residual_block_0/conv1D_1/Pad:output:0>TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Д “
BTCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpKtcn_1_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0y
7TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
3TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsJTCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0@TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  О
&TCN_1/residual_block_0/conv1D_1/Conv1DConv2D:TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims:output:0<TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingVALID*
strides
Ѕ
.TCN_1/residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze/TCN_1/residual_block_0/conv1D_1/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€≤
6TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp?tcn_1_residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0в
'TCN_1/residual_block_0/conv1D_1/BiasAddBiasAdd7TCN_1/residual_block_0/conv1D_1/Conv1D/Squeeze:output:0>TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
(TCN_1/residual_block_0/Act_Conv1D_1/ReluRelu0TCN_1/residual_block_0/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В •
*TCN_1/residual_block_0/SDropout_1/IdentityIdentity6TCN_1/residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Я
+TCN_1/residual_block_0/Act_Conv_Blocks/ReluRelu3TCN_1/residual_block_0/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В З
<TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€–
8TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputsETCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ва
ITCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpRtcn_1_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0А
>TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Х
:TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsQTCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0GTCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Ґ
-TCN_1/residual_block_0/matching_conv1D/Conv1DConv2DATCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0CTCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingSAME*
strides
ѕ
5TCN_1/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze6TCN_1/residual_block_0/matching_conv1D/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€ј
=TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOpFtcn_1_residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ч
.TCN_1/residual_block_0/matching_conv1D/BiasAddBiasAdd>TCN_1/residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0ETCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В ÷
"TCN_1/residual_block_0/Add_Res/addAddV27TCN_1/residual_block_0/matching_conv1D/BiasAdd:output:09TCN_1/residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Р
)TCN_1/residual_block_0/Act_Res_Block/ReluRelu&TCN_1/residual_block_0/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В Х
,TCN_1/residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       —
#TCN_1/residual_block_1/conv1D_0/PadPad7TCN_1/residual_block_0/Act_Res_Block/Relu:activations:05TCN_1/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ж ~
4TCN_1/residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ю
STCN_1/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ЖЃ
UTCN_1/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ©
PTCN_1/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¶
MTCN_1/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        Л
ATCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ч
>TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ј
5TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND,TCN_1/residual_block_1/conv1D_0/Pad:output:0JTCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0GTCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€C А
5TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€щ
1TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims>TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:0>TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€C “
BTCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpKtcn_1_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0y
7TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
3TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsJTCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0@TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Н
&TCN_1/residual_block_1/conv1D_0/Conv1DConv2D:TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims:output:0<TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€A *
paddingVALID*
strides
ј
.TCN_1/residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze/TCN_1/residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€A *
squeeze_dims

э€€€€€€€€Л
ATCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ф
;TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        …
5TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND7TCN_1/residual_block_1/conv1D_0/Conv1D/Squeeze:output:0JTCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0DTCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ≤
6TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp?tcn_1_residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0й
'TCN_1/residual_block_1/conv1D_0/BiasAddBiasAdd>TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:0>TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
(TCN_1/residual_block_1/Act_Conv1D_0/ReluRelu0TCN_1/residual_block_1/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В •
*TCN_1/residual_block_1/SDropout_0/IdentityIdentity6TCN_1/residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Х
,TCN_1/residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ќ
#TCN_1/residual_block_1/conv1D_1/PadPad3TCN_1/residual_block_1/SDropout_0/Identity:output:05TCN_1/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ж ~
4TCN_1/residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ю
STCN_1/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ЖЃ
UTCN_1/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ©
PTCN_1/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¶
MTCN_1/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        Л
ATCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ч
>TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ј
5TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND,TCN_1/residual_block_1/conv1D_1/Pad:output:0JTCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0GTCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€C А
5TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€щ
1TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims>TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:0>TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€C “
BTCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpKtcn_1_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0y
7TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
3TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsJTCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0@TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Н
&TCN_1/residual_block_1/conv1D_1/Conv1DConv2D:TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims:output:0<TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€A *
paddingVALID*
strides
ј
.TCN_1/residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze/TCN_1/residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€A *
squeeze_dims

э€€€€€€€€Л
ATCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ф
;TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        …
5TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND7TCN_1/residual_block_1/conv1D_1/Conv1D/Squeeze:output:0JTCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0DTCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ≤
6TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp?tcn_1_residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0й
'TCN_1/residual_block_1/conv1D_1/BiasAddBiasAdd>TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:0>TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
(TCN_1/residual_block_1/Act_Conv1D_1/ReluRelu0TCN_1/residual_block_1/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В •
*TCN_1/residual_block_1/SDropout_1/IdentityIdentity6TCN_1/residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Я
+TCN_1/residual_block_1/Act_Conv_Blocks/ReluRelu3TCN_1/residual_block_1/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ÷
"TCN_1/residual_block_1/Add_Res/addAddV27TCN_1/residual_block_0/Act_Res_Block/Relu:activations:09TCN_1/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Р
)TCN_1/residual_block_1/Act_Res_Block/ReluRelu&TCN_1/residual_block_1/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В Х
,TCN_1/residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       —
#TCN_1/residual_block_2/conv1D_0/PadPad7TCN_1/residual_block_1/Act_Res_Block/Relu:activations:05TCN_1/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€К ~
4TCN_1/residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ю
STCN_1/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:КЃ
UTCN_1/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ©
PTCN_1/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¶
MTCN_1/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Л
ATCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ч
>TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ј
5TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND,TCN_1/residual_block_2/conv1D_0/Pad:output:0JTCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0GTCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€# А
5TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€щ
1TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims>TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:0>TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€# “
BTCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpKtcn_1_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0y
7TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
3TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsJTCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0@TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Н
&TCN_1/residual_block_2/conv1D_0/Conv1DConv2D:TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims:output:0<TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€! *
paddingVALID*
strides
ј
.TCN_1/residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze/TCN_1/residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€! *
squeeze_dims

э€€€€€€€€Л
ATCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ф
;TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       …
5TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND7TCN_1/residual_block_2/conv1D_0/Conv1D/Squeeze:output:0JTCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0DTCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ≤
6TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp?tcn_1_residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0й
'TCN_1/residual_block_2/conv1D_0/BiasAddBiasAdd>TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:0>TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
(TCN_1/residual_block_2/Act_Conv1D_0/ReluRelu0TCN_1/residual_block_2/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В •
*TCN_1/residual_block_2/SDropout_0/IdentityIdentity6TCN_1/residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Х
,TCN_1/residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ќ
#TCN_1/residual_block_2/conv1D_1/PadPad3TCN_1/residual_block_2/SDropout_0/Identity:output:05TCN_1/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€К ~
4TCN_1/residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ю
STCN_1/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:КЃ
UTCN_1/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ©
PTCN_1/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¶
MTCN_1/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Л
ATCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ч
>TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ј
5TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND,TCN_1/residual_block_2/conv1D_1/Pad:output:0JTCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0GTCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€# А
5TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€щ
1TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims>TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:0>TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€# “
BTCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpKtcn_1_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0y
7TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
3TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsJTCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0@TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Н
&TCN_1/residual_block_2/conv1D_1/Conv1DConv2D:TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims:output:0<TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€! *
paddingVALID*
strides
ј
.TCN_1/residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze/TCN_1/residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€! *
squeeze_dims

э€€€€€€€€Л
ATCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ф
;TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       …
5TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND7TCN_1/residual_block_2/conv1D_1/Conv1D/Squeeze:output:0JTCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0DTCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ≤
6TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp?tcn_1_residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0й
'TCN_1/residual_block_2/conv1D_1/BiasAddBiasAdd>TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:0>TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
(TCN_1/residual_block_2/Act_Conv1D_1/ReluRelu0TCN_1/residual_block_2/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В •
*TCN_1/residual_block_2/SDropout_1/IdentityIdentity6TCN_1/residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Я
+TCN_1/residual_block_2/Act_Conv_Blocks/ReluRelu3TCN_1/residual_block_2/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ÷
"TCN_1/residual_block_2/Add_Res/addAddV27TCN_1/residual_block_1/Act_Res_Block/Relu:activations:09TCN_1/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Р
)TCN_1/residual_block_2/Act_Res_Block/ReluRelu&TCN_1/residual_block_2/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В Х
,TCN_1/residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       —
#TCN_1/residual_block_3/conv1D_0/PadPad7TCN_1/residual_block_2/Act_Res_Block/Relu:activations:05TCN_1/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Т ~
4TCN_1/residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ю
STCN_1/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ТЃ
UTCN_1/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ©
PTCN_1/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¶
MTCN_1/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Л
ATCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ч
>TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ј
5TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND,TCN_1/residual_block_3/conv1D_0/Pad:output:0JTCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0GTCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€ А
5TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€щ
1TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims>TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:0>TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ “
BTCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpKtcn_1_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0y
7TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
3TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsJTCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0@TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Н
&TCN_1/residual_block_3/conv1D_0/Conv1DConv2D:TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims:output:0<TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
ј
.TCN_1/residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze/TCN_1/residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€Л
ATCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ф
;TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       …
5TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND7TCN_1/residual_block_3/conv1D_0/Conv1D/Squeeze:output:0JTCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0DTCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ≤
6TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp?tcn_1_residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0й
'TCN_1/residual_block_3/conv1D_0/BiasAddBiasAdd>TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:0>TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
(TCN_1/residual_block_3/Act_Conv1D_0/ReluRelu0TCN_1/residual_block_3/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В •
*TCN_1/residual_block_3/SDropout_0/IdentityIdentity6TCN_1/residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Х
,TCN_1/residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ќ
#TCN_1/residual_block_3/conv1D_1/PadPad3TCN_1/residual_block_3/SDropout_0/Identity:output:05TCN_1/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Т ~
4TCN_1/residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ю
STCN_1/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ТЃ
UTCN_1/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ©
PTCN_1/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¶
MTCN_1/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Л
ATCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ч
>TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ј
5TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND,TCN_1/residual_block_3/conv1D_1/Pad:output:0JTCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0GTCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€ А
5TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€щ
1TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims>TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:0>TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ “
BTCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpKtcn_1_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0y
7TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
3TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsJTCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0@TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Н
&TCN_1/residual_block_3/conv1D_1/Conv1DConv2D:TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims:output:0<TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
ј
.TCN_1/residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze/TCN_1/residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€Л
ATCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ф
;TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       …
5TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND7TCN_1/residual_block_3/conv1D_1/Conv1D/Squeeze:output:0JTCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0DTCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ≤
6TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp?tcn_1_residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0й
'TCN_1/residual_block_3/conv1D_1/BiasAddBiasAdd>TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:0>TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
(TCN_1/residual_block_3/Act_Conv1D_1/ReluRelu0TCN_1/residual_block_3/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В •
*TCN_1/residual_block_3/SDropout_1/IdentityIdentity6TCN_1/residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Я
+TCN_1/residual_block_3/Act_Conv_Blocks/ReluRelu3TCN_1/residual_block_3/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ÷
"TCN_1/residual_block_3/Add_Res/addAddV27TCN_1/residual_block_2/Act_Res_Block/Relu:activations:09TCN_1/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Р
)TCN_1/residual_block_3/Act_Res_Block/ReluRelu&TCN_1/residual_block_3/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В ‘
TCN_1/Add_Skip_Connections/addAddV29TCN_1/residual_block_0/Act_Conv_Blocks/Relu:activations:09TCN_1/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В њ
 TCN_1/Add_Skip_Connections/add_1AddV2"TCN_1/Add_Skip_Connections/add:z:09TCN_1/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Ѕ
 TCN_1/Add_Skip_Connections/add_2AddV2$TCN_1/Add_Skip_Connections/add_1:z:09TCN_1/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В {
&TCN_1/Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    €€€€    }
(TCN_1/Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            }
(TCN_1/Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ж
 TCN_1/Slice_Output/strided_sliceStridedSlice$TCN_1/Add_Skip_Connections/add_2:z:0/TCN_1/Slice_Output/strided_slice/stack:output:01TCN_1/Slice_Output/strided_slice/stack_1:output:01TCN_1/Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *

begin_mask*
end_mask*
shrink_axis_maskД
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ь
Dense_1/MatMulMatMul)TCN_1/Slice_Output/strided_slice:output:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ В
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ `
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Д
Dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Н
Dense_2/MatMulMatMulDense_1/Relu:activations:0%Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ В
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
Dense_2/BiasAddBiasAddDense_2/MatMul:product:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ `
Dense_2/ReluReluDense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ В
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Л
Output/MatMulMatMulDense_2/Relu:activations:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€А
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ч
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ж
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0*Dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Т
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0В
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0(Dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ч
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ж
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
Dense_2/kernel/Regularizer/mulMul)Dense_2/kernel/Regularizer/mul/x:output:0*Dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Т
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0В
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
Dense_2/bias/Regularizer/mulMul'Dense_2/bias/Regularizer/mul/x:output:0(Dense_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Х
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ъ
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Р
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0А
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ф
Output/bias/Regularizer/mulMul&Output/bias/Regularizer/mul/x:output:0'Output/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityOutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≠
NoOpNoOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp/^Dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp^Dense_2/MatMul/ReadVariableOp/^Dense_2/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp.^Output/bias/Regularizer/L2Loss/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp7^TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpC^TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp7^TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpC^TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp>^TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpJ^TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp7^TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpC^TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp7^TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpC^TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp7^TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpC^TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp7^TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpC^TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp7^TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpC^TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp7^TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpC^TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€В: : : : : : : : : : : : : : : : : : : : : : : : 2@
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
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp2p
6TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp6TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2И
BTCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpBTCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2p
6TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp6TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2И
BTCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpBTCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2~
=TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp=TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2Ц
ITCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpITCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2p
6TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp6TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2И
BTCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpBTCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2p
6TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp6TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2И
BTCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpBTCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2p
6TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp6TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2И
BTCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpBTCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2p
6TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp6TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2И
BTCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpBTCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2p
6TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp6TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2И
BTCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpBTCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2p
6TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp6TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2И
BTCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpBTCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€В
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_13009

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
к∆
”
@__inference_TCN_1_layer_call_and_return_conditional_losses_13315

inputs[
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource: G
9residual_block_0_conv1d_0_biasadd_readvariableop_resource: [
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_0_conv1d_1_biasadd_readvariableop_resource: b
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource: N
@residual_block_0_matching_conv1d_biasadd_readvariableop_resource: [
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_1_conv1d_0_biasadd_readvariableop_resource: [
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_1_conv1d_1_biasadd_readvariableop_resource: [
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_2_conv1d_0_biasadd_readvariableop_resource: [
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_2_conv1d_1_biasadd_readvariableop_resource: [
Eresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_3_conv1d_0_biasadd_readvariableop_resource: [
Eresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_3_conv1d_1_biasadd_readvariableop_resource: 
identityИҐ0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpҐ<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpҐ<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpҐCresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpҐ<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpҐ<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpҐ<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpҐ<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpҐ<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpҐ<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpП
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ф
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Дz
/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€÷
+residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_0/Pad:output:08residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Д∆
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0s
1residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ь
 residual_block_0/conv1D_0/Conv1DConv2D4residual_block_0/conv1D_0/Conv1D/ExpandDims:output:06residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingVALID*
strides
µ
(residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_0/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€¶
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0–
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/Conv1D/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_0/Act_Conv1D_0/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_0/SDropout_0/IdentityIdentity0residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ї
residual_block_0/conv1D_1/PadPad-residual_block_0/SDropout_0/Identity:output:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Д z
/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€÷
+residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_1/Pad:output:08residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Д ∆
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ь
 residual_block_0/conv1D_1/Conv1DConv2D4residual_block_0/conv1D_1/Conv1D/ExpandDims:output:06residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingVALID*
strides
µ
(residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_1/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€¶
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0–
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/Conv1D/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_0/Act_Conv1D_1/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_0/SDropout_1/IdentityIdentity0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В У
%residual_block_0/Act_Conv_Blocks/ReluRelu-residual_block_0/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Б
6residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€ƒ
2residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputs?residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€В‘
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpLresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0z
8residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Г
4residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsKresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Aresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Р
'residual_block_0/matching_conv1D/Conv1DConv2D;residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0=residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingSAME*
strides
√
/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze0residual_block_0/matching_conv1D/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€і
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0е
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В ƒ
residual_block_0/Add_Res/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:03residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Д
#residual_block_0/Act_Res_Block/ReluRelu residual_block_0/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       њ
residual_block_1/conv1D_0/PadPad1residual_block_0/Act_Res_Block/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ж x
.residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ж®
Oresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        †
Gresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        Е
;residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ®
/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_0/Pad:output:0Dresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€C z
/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€C ∆
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_1/conv1D_0/Conv1DConv2D4residual_block_1/conv1D_0/Conv1D/ExpandDims:output:06residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€A *
paddingVALID*
strides
і
(residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€A *
squeeze_dims

э€€€€€€€€Е
;residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ±
/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_1/Act_Conv1D_0/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_1/SDropout_0/IdentityIdentity0residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ї
residual_block_1/conv1D_1/PadPad-residual_block_1/SDropout_0/Identity:output:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ж x
.residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ж®
Oresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        †
Gresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        Е
;residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ®
/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_1/Pad:output:0Dresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€C z
/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€C ∆
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_1/conv1D_1/Conv1DConv2D4residual_block_1/conv1D_1/Conv1D/ExpandDims:output:06residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€A *
paddingVALID*
strides
і
(residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€A *
squeeze_dims

э€€€€€€€€Е
;residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ±
/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_1/Act_Conv1D_1/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_1/SDropout_1/IdentityIdentity0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В У
%residual_block_1/Act_Conv_Blocks/ReluRelu-residual_block_1/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ƒ
residual_block_1/Add_Res/addAddV21residual_block_0/Act_Res_Block/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Д
#residual_block_1/Act_Res_Block/ReluRelu residual_block_1/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       њ
residual_block_2/conv1D_0/PadPad1residual_block_1/Act_Res_Block/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€К x
.residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:К®
Oresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       †
Gresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Е
;residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ®
/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_0/Pad:output:0Dresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€# z
/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€# ∆
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_2/conv1D_0/Conv1DConv2D4residual_block_2/conv1D_0/Conv1D/ExpandDims:output:06residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€! *
paddingVALID*
strides
і
(residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€! *
squeeze_dims

э€€€€€€€€Е
;residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ±
/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_2/Act_Conv1D_0/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_2/SDropout_0/IdentityIdentity0residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ї
residual_block_2/conv1D_1/PadPad-residual_block_2/SDropout_0/Identity:output:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€К x
.residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:К®
Oresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       †
Gresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Е
;residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ®
/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_1/Pad:output:0Dresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€# z
/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€# ∆
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_2/conv1D_1/Conv1DConv2D4residual_block_2/conv1D_1/Conv1D/ExpandDims:output:06residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€! *
paddingVALID*
strides
і
(residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€! *
squeeze_dims

э€€€€€€€€Е
;residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ±
/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_2/Act_Conv1D_1/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_2/SDropout_1/IdentityIdentity0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В У
%residual_block_2/Act_Conv_Blocks/ReluRelu-residual_block_2/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ƒ
residual_block_2/Add_Res/addAddV21residual_block_1/Act_Res_Block/Relu:activations:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Д
#residual_block_2/Act_Res_Block/ReluRelu residual_block_2/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       њ
residual_block_3/conv1D_0/PadPad1residual_block_2/Act_Res_Block/Relu:activations:0/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Т x
.residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Т®
Oresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       †
Gresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Е
;residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ®
/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_0/Pad:output:0Dresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€ z
/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ∆
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_3/conv1D_0/Conv1DConv2D4residual_block_3/conv1D_0/Conv1D/ExpandDims:output:06residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
і
(residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€Е
;residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ±
/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_3/conv1D_0/BiasAddBiasAdd8residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_3/Act_Conv1D_0/ReluRelu*residual_block_3/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_3/SDropout_0/IdentityIdentity0residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ї
residual_block_3/conv1D_1/PadPad-residual_block_3/SDropout_0/Identity:output:0/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Т x
.residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Т®
Oresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       †
Gresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Е
;residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ®
/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_1/Pad:output:0Dresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€ z
/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ∆
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_3/conv1D_1/Conv1DConv2D4residual_block_3/conv1D_1/Conv1D/ExpandDims:output:06residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
і
(residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€Е
;residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ±
/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_3/conv1D_1/BiasAddBiasAdd8residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_3/Act_Conv1D_1/ReluRelu*residual_block_3/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_3/SDropout_1/IdentityIdentity0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В У
%residual_block_3/Act_Conv_Blocks/ReluRelu-residual_block_3/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ƒ
residual_block_3/Add_Res/addAddV21residual_block_2/Act_Res_Block/Relu:activations:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Д
#residual_block_3/Act_Res_Block/ReluRelu residual_block_3/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В ¬
Add_Skip_Connections/addAddV23residual_block_0/Act_Conv_Blocks/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В ≠
Add_Skip_Connections/add_1AddV2Add_Skip_Connections/add:z:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В ѓ
Add_Skip_Connections/add_2AddV2Add_Skip_Connections/add_1:z:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В u
 Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    €€€€    w
"Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            w
"Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         »
Slice_Output/strided_sliceStridedSliceAdd_Skip_Connections/add_2:z:0)Slice_Output/strided_slice/stack:output:0+Slice_Output/strided_slice/stack_1:output:0+Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *

begin_mask*
end_mask*
shrink_axis_maskr
IdentityIdentity#Slice_Output/strided_slice:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ÷
NoOpNoOp1^residual_block_0/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_0/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp8^residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpD^residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€В: : : : : : : : : : : : : : : : : : 2d
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2r
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2К
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpCresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€В
 
_user_specified_nameinputs
Љ
У
&__inference_Output_layer_call_fn_15512

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_Output_layer_call_and_return_conditional_losses_13422o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Џ=
ы	
G__inference_sequential_1_layer_call_and_return_conditional_losses_13969

inputs!
tcn_1_13892: 
tcn_1_13894: !
tcn_1_13896:  
tcn_1_13898: !
tcn_1_13900: 
tcn_1_13902: !
tcn_1_13904:  
tcn_1_13906: !
tcn_1_13908:  
tcn_1_13910: !
tcn_1_13912:  
tcn_1_13914: !
tcn_1_13916:  
tcn_1_13918: !
tcn_1_13920:  
tcn_1_13922: !
tcn_1_13924:  
tcn_1_13926: 
dense_1_13929:  
dense_1_13931: 
dense_2_13934:  
dense_2_13936: 
output_13939: 
output_13941:
identityИҐDense_1/StatefulPartitionedCallҐ.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpҐDense_2/StatefulPartitionedCallҐ.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpҐ0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpҐOutput/StatefulPartitionedCallҐ-Output/bias/Regularizer/L2Loss/ReadVariableOpҐ/Output/kernel/Regularizer/L2Loss/ReadVariableOpҐTCN_1/StatefulPartitionedCall—
TCN_1/StatefulPartitionedCallStatefulPartitionedCallinputstcn_1_13892tcn_1_13894tcn_1_13896tcn_1_13898tcn_1_13900tcn_1_13902tcn_1_13904tcn_1_13906tcn_1_13908tcn_1_13910tcn_1_13912tcn_1_13914tcn_1_13916tcn_1_13918tcn_1_13920tcn_1_13922tcn_1_13924tcn_1_13926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_TCN_1_layer_call_and_return_conditional_losses_13675Й
Dense_1/StatefulPartitionedCallStatefulPartitionedCall&TCN_1/StatefulPartitionedCall:output:0dense_1_13929dense_1_13931*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_13372Л
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_13934dense_2_13936*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_13397З
Output/StatefulPartitionedCallStatefulPartitionedCall(Dense_2/StatefulPartitionedCall:output:0output_13939output_13941*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_Output_layer_call_and_return_conditional_losses_13422~
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_13929*
_output_shapes

:  *
dtype0Ж
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0*Dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_13931*
_output_shapes
: *
dtype0В
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0(Dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ~
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_13934*
_output_shapes

:  *
dtype0Ж
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
Dense_2/kernel/Regularizer/mulMul)Dense_2/kernel/Regularizer/mul/x:output:0*Dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_13936*
_output_shapes
: *
dtype0В
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
Dense_2/bias/Regularizer/mulMul'Dense_2/bias/Regularizer/mul/x:output:0(Dense_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_13939*
_output_shapes

: *
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ъ
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_13941*
_output_shapes
:*
dtype0А
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ф
Output/bias/Regularizer/mulMul&Output/bias/Regularizer/mul/x:output:0'Output/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€х
NoOpNoOp ^Dense_1/StatefulPartitionedCall/^Dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^Dense_2/StatefulPartitionedCall/^Dense_2/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp^Output/StatefulPartitionedCall.^Output/bias/Regularizer/L2Loss/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp^TCN_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€В: : : : : : : : : : : : : : : : : : : : : : : : 2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2`
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2`
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2^
-Output/bias/Regularizer/L2Loss/ReadVariableOp-Output/bias/Regularizer/L2Loss/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp2>
TCN_1/StatefulPartitionedCallTCN_1/StatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€В
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_12943

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15700

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ц
н
%__inference_TCN_1_layer_call_fn_15007

inputs
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9:  

unknown_10:  

unknown_11:  

unknown_12:  

unknown_13:  

unknown_14:  

unknown_15:  

unknown_16: 
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_TCN_1_layer_call_and_return_conditional_losses_13675o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€В: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€В
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15680

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ё
„
B__inference_Dense_2_layer_call_and_return_conditional_losses_13397

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐ.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpҐ0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ П
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ж
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
Dense_2/kernel/Regularizer/mulMul)Dense_2/kernel/Regularizer/mul/x:output:0*Dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: К
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0В
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
Dense_2/bias/Regularizer/mulMul'Dense_2/bias/Regularizer/mul/x:output:0(Dense_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ џ
NoOpNoOp^BiasAdd/ReadVariableOp/^Dense_2/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2`
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
к∆
”
@__inference_TCN_1_layer_call_and_return_conditional_losses_15447

inputs[
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource: G
9residual_block_0_conv1d_0_biasadd_readvariableop_resource: [
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_0_conv1d_1_biasadd_readvariableop_resource: b
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource: N
@residual_block_0_matching_conv1d_biasadd_readvariableop_resource: [
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_1_conv1d_0_biasadd_readvariableop_resource: [
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_1_conv1d_1_biasadd_readvariableop_resource: [
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_2_conv1d_0_biasadd_readvariableop_resource: [
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_2_conv1d_1_biasadd_readvariableop_resource: [
Eresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_3_conv1d_0_biasadd_readvariableop_resource: [
Eresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_3_conv1d_1_biasadd_readvariableop_resource: 
identityИҐ0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpҐ<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpҐ<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpҐCresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpҐ<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpҐ<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpҐ<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpҐ<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpҐ<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpҐ<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpП
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ф
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Дz
/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€÷
+residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_0/Pad:output:08residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Д∆
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0s
1residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ь
 residual_block_0/conv1D_0/Conv1DConv2D4residual_block_0/conv1D_0/Conv1D/ExpandDims:output:06residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingVALID*
strides
µ
(residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_0/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€¶
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0–
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/Conv1D/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_0/Act_Conv1D_0/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_0/SDropout_0/IdentityIdentity0residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ї
residual_block_0/conv1D_1/PadPad-residual_block_0/SDropout_0/Identity:output:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Д z
/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€÷
+residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_1/Pad:output:08residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Д ∆
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ь
 residual_block_0/conv1D_1/Conv1DConv2D4residual_block_0/conv1D_1/Conv1D/ExpandDims:output:06residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingVALID*
strides
µ
(residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_1/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€¶
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0–
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/Conv1D/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_0/Act_Conv1D_1/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_0/SDropout_1/IdentityIdentity0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В У
%residual_block_0/Act_Conv_Blocks/ReluRelu-residual_block_0/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Б
6residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€ƒ
2residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputs?residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€В‘
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpLresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0z
8residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Г
4residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsKresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Aresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Р
'residual_block_0/matching_conv1D/Conv1DConv2D;residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0=residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingSAME*
strides
√
/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze0residual_block_0/matching_conv1D/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€і
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0е
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В ƒ
residual_block_0/Add_Res/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:03residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Д
#residual_block_0/Act_Res_Block/ReluRelu residual_block_0/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       њ
residual_block_1/conv1D_0/PadPad1residual_block_0/Act_Res_Block/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ж x
.residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ж®
Oresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        †
Gresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        Е
;residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ®
/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_0/Pad:output:0Dresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€C z
/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€C ∆
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_1/conv1D_0/Conv1DConv2D4residual_block_1/conv1D_0/Conv1D/ExpandDims:output:06residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€A *
paddingVALID*
strides
і
(residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€A *
squeeze_dims

э€€€€€€€€Е
;residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ±
/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_1/Act_Conv1D_0/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_1/SDropout_0/IdentityIdentity0residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ї
residual_block_1/conv1D_1/PadPad-residual_block_1/SDropout_0/Identity:output:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ж x
.residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ж®
Oresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        †
Gresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        Е
;residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ®
/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_1/Pad:output:0Dresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€C z
/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€C ∆
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_1/conv1D_1/Conv1DConv2D4residual_block_1/conv1D_1/Conv1D/ExpandDims:output:06residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€A *
paddingVALID*
strides
і
(residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€A *
squeeze_dims

э€€€€€€€€Е
;residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ±
/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_1/Act_Conv1D_1/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_1/SDropout_1/IdentityIdentity0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В У
%residual_block_1/Act_Conv_Blocks/ReluRelu-residual_block_1/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ƒ
residual_block_1/Add_Res/addAddV21residual_block_0/Act_Res_Block/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Д
#residual_block_1/Act_Res_Block/ReluRelu residual_block_1/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       њ
residual_block_2/conv1D_0/PadPad1residual_block_1/Act_Res_Block/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€К x
.residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:К®
Oresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       †
Gresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Е
;residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ®
/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_0/Pad:output:0Dresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€# z
/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€# ∆
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_2/conv1D_0/Conv1DConv2D4residual_block_2/conv1D_0/Conv1D/ExpandDims:output:06residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€! *
paddingVALID*
strides
і
(residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€! *
squeeze_dims

э€€€€€€€€Е
;residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ±
/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_2/Act_Conv1D_0/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_2/SDropout_0/IdentityIdentity0residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ї
residual_block_2/conv1D_1/PadPad-residual_block_2/SDropout_0/Identity:output:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€К x
.residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:К®
Oresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       †
Gresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Е
;residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ®
/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_1/Pad:output:0Dresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€# z
/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€# ∆
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_2/conv1D_1/Conv1DConv2D4residual_block_2/conv1D_1/Conv1D/ExpandDims:output:06residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€! *
paddingVALID*
strides
і
(residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€! *
squeeze_dims

э€€€€€€€€Е
;residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ±
/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_2/Act_Conv1D_1/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_2/SDropout_1/IdentityIdentity0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В У
%residual_block_2/Act_Conv_Blocks/ReluRelu-residual_block_2/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ƒ
residual_block_2/Add_Res/addAddV21residual_block_1/Act_Res_Block/Relu:activations:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Д
#residual_block_2/Act_Res_Block/ReluRelu residual_block_2/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       њ
residual_block_3/conv1D_0/PadPad1residual_block_2/Act_Res_Block/Relu:activations:0/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Т x
.residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Т®
Oresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       †
Gresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Е
;residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ®
/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_0/Pad:output:0Dresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€ z
/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ∆
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_3/conv1D_0/Conv1DConv2D4residual_block_3/conv1D_0/Conv1D/ExpandDims:output:06residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
і
(residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€Е
;residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ±
/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_3/conv1D_0/BiasAddBiasAdd8residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_3/Act_Conv1D_0/ReluRelu*residual_block_3/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_3/SDropout_0/IdentityIdentity0residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ї
residual_block_3/conv1D_1/PadPad-residual_block_3/SDropout_0/Identity:output:0/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Т x
.residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Т®
Oresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       †
Gresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Е
;residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ®
/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_1/Pad:output:0Dresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€ z
/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ∆
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_3/conv1D_1/Conv1DConv2D4residual_block_3/conv1D_1/Conv1D/ExpandDims:output:06residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
і
(residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€Е
;residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ±
/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_3/conv1D_1/BiasAddBiasAdd8residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_3/Act_Conv1D_1/ReluRelu*residual_block_3/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_3/SDropout_1/IdentityIdentity0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В У
%residual_block_3/Act_Conv_Blocks/ReluRelu-residual_block_3/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ƒ
residual_block_3/Add_Res/addAddV21residual_block_2/Act_Res_Block/Relu:activations:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Д
#residual_block_3/Act_Res_Block/ReluRelu residual_block_3/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В ¬
Add_Skip_Connections/addAddV23residual_block_0/Act_Conv_Blocks/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В ≠
Add_Skip_Connections/add_1AddV2Add_Skip_Connections/add:z:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В ѓ
Add_Skip_Connections/add_2AddV2Add_Skip_Connections/add_1:z:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В u
 Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    €€€€    w
"Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            w
"Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         »
Slice_Output/strided_sliceStridedSliceAdd_Skip_Connections/add_2:z:0)Slice_Output/strided_slice/stack:output:0+Slice_Output/strided_slice/stack_1:output:0+Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *

begin_mask*
end_mask*
shrink_axis_maskr
IdentityIdentity#Slice_Output/strided_slice:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ÷
NoOpNoOp1^residual_block_0/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_0/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp8^residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpD^residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€В: : : : : : : : : : : : : : : : : : 2d
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2r
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2К
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpCresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€В
 
_user_specified_nameinputs
ч
F
*__inference_SDropout_0_layer_call_fn_15715

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_13058v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_12992

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_13014

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
F
*__inference_SDropout_1_layer_call_fn_15615

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_12948v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
F
*__inference_SDropout_0_layer_call_fn_15590

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_12921v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
F
*__inference_SDropout_1_layer_call_fn_15655

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_12992v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15660

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
у‘
Ъ:
!__inference__traced_restore_16480
file_prefix1
assignvariableop_dense_1_kernel:  -
assignvariableop_1_dense_1_bias: 3
!assignvariableop_2_dense_2_kernel:  -
assignvariableop_3_dense_2_bias: 2
 assignvariableop_4_output_kernel: ,
assignvariableop_5_output_bias:O
9assignvariableop_6_tcn_1_residual_block_0_conv1d_0_kernel: E
7assignvariableop_7_tcn_1_residual_block_0_conv1d_0_bias: O
9assignvariableop_8_tcn_1_residual_block_0_conv1d_1_kernel:  E
7assignvariableop_9_tcn_1_residual_block_0_conv1d_1_bias: W
Aassignvariableop_10_tcn_1_residual_block_0_matching_conv1d_kernel: M
?assignvariableop_11_tcn_1_residual_block_0_matching_conv1d_bias: P
:assignvariableop_12_tcn_1_residual_block_1_conv1d_0_kernel:  F
8assignvariableop_13_tcn_1_residual_block_1_conv1d_0_bias: P
:assignvariableop_14_tcn_1_residual_block_1_conv1d_1_kernel:  F
8assignvariableop_15_tcn_1_residual_block_1_conv1d_1_bias: P
:assignvariableop_16_tcn_1_residual_block_2_conv1d_0_kernel:  F
8assignvariableop_17_tcn_1_residual_block_2_conv1d_0_bias: P
:assignvariableop_18_tcn_1_residual_block_2_conv1d_1_kernel:  F
8assignvariableop_19_tcn_1_residual_block_2_conv1d_1_bias: P
:assignvariableop_20_tcn_1_residual_block_3_conv1d_0_kernel:  F
8assignvariableop_21_tcn_1_residual_block_3_conv1d_0_bias: P
:assignvariableop_22_tcn_1_residual_block_3_conv1d_1_kernel:  F
8assignvariableop_23_tcn_1_residual_block_3_conv1d_1_bias: '
assignvariableop_24_iteration:	 +
!assignvariableop_25_learning_rate: W
Aassignvariableop_26_adam_m_tcn_1_residual_block_0_conv1d_0_kernel: W
Aassignvariableop_27_adam_v_tcn_1_residual_block_0_conv1d_0_kernel: M
?assignvariableop_28_adam_m_tcn_1_residual_block_0_conv1d_0_bias: M
?assignvariableop_29_adam_v_tcn_1_residual_block_0_conv1d_0_bias: W
Aassignvariableop_30_adam_m_tcn_1_residual_block_0_conv1d_1_kernel:  W
Aassignvariableop_31_adam_v_tcn_1_residual_block_0_conv1d_1_kernel:  M
?assignvariableop_32_adam_m_tcn_1_residual_block_0_conv1d_1_bias: M
?assignvariableop_33_adam_v_tcn_1_residual_block_0_conv1d_1_bias: ^
Hassignvariableop_34_adam_m_tcn_1_residual_block_0_matching_conv1d_kernel: ^
Hassignvariableop_35_adam_v_tcn_1_residual_block_0_matching_conv1d_kernel: T
Fassignvariableop_36_adam_m_tcn_1_residual_block_0_matching_conv1d_bias: T
Fassignvariableop_37_adam_v_tcn_1_residual_block_0_matching_conv1d_bias: W
Aassignvariableop_38_adam_m_tcn_1_residual_block_1_conv1d_0_kernel:  W
Aassignvariableop_39_adam_v_tcn_1_residual_block_1_conv1d_0_kernel:  M
?assignvariableop_40_adam_m_tcn_1_residual_block_1_conv1d_0_bias: M
?assignvariableop_41_adam_v_tcn_1_residual_block_1_conv1d_0_bias: W
Aassignvariableop_42_adam_m_tcn_1_residual_block_1_conv1d_1_kernel:  W
Aassignvariableop_43_adam_v_tcn_1_residual_block_1_conv1d_1_kernel:  M
?assignvariableop_44_adam_m_tcn_1_residual_block_1_conv1d_1_bias: M
?assignvariableop_45_adam_v_tcn_1_residual_block_1_conv1d_1_bias: W
Aassignvariableop_46_adam_m_tcn_1_residual_block_2_conv1d_0_kernel:  W
Aassignvariableop_47_adam_v_tcn_1_residual_block_2_conv1d_0_kernel:  M
?assignvariableop_48_adam_m_tcn_1_residual_block_2_conv1d_0_bias: M
?assignvariableop_49_adam_v_tcn_1_residual_block_2_conv1d_0_bias: W
Aassignvariableop_50_adam_m_tcn_1_residual_block_2_conv1d_1_kernel:  W
Aassignvariableop_51_adam_v_tcn_1_residual_block_2_conv1d_1_kernel:  M
?assignvariableop_52_adam_m_tcn_1_residual_block_2_conv1d_1_bias: M
?assignvariableop_53_adam_v_tcn_1_residual_block_2_conv1d_1_bias: W
Aassignvariableop_54_adam_m_tcn_1_residual_block_3_conv1d_0_kernel:  W
Aassignvariableop_55_adam_v_tcn_1_residual_block_3_conv1d_0_kernel:  M
?assignvariableop_56_adam_m_tcn_1_residual_block_3_conv1d_0_bias: M
?assignvariableop_57_adam_v_tcn_1_residual_block_3_conv1d_0_bias: W
Aassignvariableop_58_adam_m_tcn_1_residual_block_3_conv1d_1_kernel:  W
Aassignvariableop_59_adam_v_tcn_1_residual_block_3_conv1d_1_kernel:  M
?assignvariableop_60_adam_m_tcn_1_residual_block_3_conv1d_1_bias: M
?assignvariableop_61_adam_v_tcn_1_residual_block_3_conv1d_1_bias: ;
)assignvariableop_62_adam_m_dense_1_kernel:  ;
)assignvariableop_63_adam_v_dense_1_kernel:  5
'assignvariableop_64_adam_m_dense_1_bias: 5
'assignvariableop_65_adam_v_dense_1_bias: ;
)assignvariableop_66_adam_m_dense_2_kernel:  ;
)assignvariableop_67_adam_v_dense_2_kernel:  5
'assignvariableop_68_adam_m_dense_2_bias: 5
'assignvariableop_69_adam_v_dense_2_bias: :
(assignvariableop_70_adam_m_output_kernel: :
(assignvariableop_71_adam_v_output_kernel: 4
&assignvariableop_72_adam_m_output_bias:4
&assignvariableop_73_adam_v_output_bias:%
assignvariableop_74_total_1: %
assignvariableop_75_count_1: #
assignvariableop_76_total: #
assignvariableop_77_count: 
identity_79ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_75ҐAssignVariableOp_76ҐAssignVariableOp_77ҐAssignVariableOp_8ҐAssignVariableOp_9•
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*Ћ
valueЅBЊOB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHС
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*≥
value©B¶OB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ђ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*“
_output_shapesњ
Љ:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*]
dtypesS
Q2O	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_2_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_2_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_4AssignVariableOp assignvariableop_4_output_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_5AssignVariableOpassignvariableop_5_output_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:–
AssignVariableOp_6AssignVariableOp9assignvariableop_6_tcn_1_residual_block_0_conv1d_0_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_7AssignVariableOp7assignvariableop_7_tcn_1_residual_block_0_conv1d_0_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:–
AssignVariableOp_8AssignVariableOp9assignvariableop_8_tcn_1_residual_block_0_conv1d_1_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_9AssignVariableOp7assignvariableop_9_tcn_1_residual_block_0_conv1d_1_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_10AssignVariableOpAassignvariableop_10_tcn_1_residual_block_0_matching_conv1d_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_11AssignVariableOp?assignvariableop_11_tcn_1_residual_block_0_matching_conv1d_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_12AssignVariableOp:assignvariableop_12_tcn_1_residual_block_1_conv1d_0_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_13AssignVariableOp8assignvariableop_13_tcn_1_residual_block_1_conv1d_0_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_14AssignVariableOp:assignvariableop_14_tcn_1_residual_block_1_conv1d_1_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_15AssignVariableOp8assignvariableop_15_tcn_1_residual_block_1_conv1d_1_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_16AssignVariableOp:assignvariableop_16_tcn_1_residual_block_2_conv1d_0_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_17AssignVariableOp8assignvariableop_17_tcn_1_residual_block_2_conv1d_0_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_18AssignVariableOp:assignvariableop_18_tcn_1_residual_block_2_conv1d_1_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_19AssignVariableOp8assignvariableop_19_tcn_1_residual_block_2_conv1d_1_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_20AssignVariableOp:assignvariableop_20_tcn_1_residual_block_3_conv1d_0_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_21AssignVariableOp8assignvariableop_21_tcn_1_residual_block_3_conv1d_0_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_22AssignVariableOp:assignvariableop_22_tcn_1_residual_block_3_conv1d_1_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_23AssignVariableOp8assignvariableop_23_tcn_1_residual_block_3_conv1d_1_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:ґ
AssignVariableOp_24AssignVariableOpassignvariableop_24_iterationIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_25AssignVariableOp!assignvariableop_25_learning_rateIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_26AssignVariableOpAassignvariableop_26_adam_m_tcn_1_residual_block_0_conv1d_0_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_27AssignVariableOpAassignvariableop_27_adam_v_tcn_1_residual_block_0_conv1d_0_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_28AssignVariableOp?assignvariableop_28_adam_m_tcn_1_residual_block_0_conv1d_0_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_29AssignVariableOp?assignvariableop_29_adam_v_tcn_1_residual_block_0_conv1d_0_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_30AssignVariableOpAassignvariableop_30_adam_m_tcn_1_residual_block_0_conv1d_1_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_31AssignVariableOpAassignvariableop_31_adam_v_tcn_1_residual_block_0_conv1d_1_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_32AssignVariableOp?assignvariableop_32_adam_m_tcn_1_residual_block_0_conv1d_1_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_33AssignVariableOp?assignvariableop_33_adam_v_tcn_1_residual_block_0_conv1d_1_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_34AssignVariableOpHassignvariableop_34_adam_m_tcn_1_residual_block_0_matching_conv1d_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_35AssignVariableOpHassignvariableop_35_adam_v_tcn_1_residual_block_0_matching_conv1d_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:я
AssignVariableOp_36AssignVariableOpFassignvariableop_36_adam_m_tcn_1_residual_block_0_matching_conv1d_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:я
AssignVariableOp_37AssignVariableOpFassignvariableop_37_adam_v_tcn_1_residual_block_0_matching_conv1d_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_38AssignVariableOpAassignvariableop_38_adam_m_tcn_1_residual_block_1_conv1d_0_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_39AssignVariableOpAassignvariableop_39_adam_v_tcn_1_residual_block_1_conv1d_0_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_40AssignVariableOp?assignvariableop_40_adam_m_tcn_1_residual_block_1_conv1d_0_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_41AssignVariableOp?assignvariableop_41_adam_v_tcn_1_residual_block_1_conv1d_0_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_42AssignVariableOpAassignvariableop_42_adam_m_tcn_1_residual_block_1_conv1d_1_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_43AssignVariableOpAassignvariableop_43_adam_v_tcn_1_residual_block_1_conv1d_1_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_44AssignVariableOp?assignvariableop_44_adam_m_tcn_1_residual_block_1_conv1d_1_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_45AssignVariableOp?assignvariableop_45_adam_v_tcn_1_residual_block_1_conv1d_1_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_46AssignVariableOpAassignvariableop_46_adam_m_tcn_1_residual_block_2_conv1d_0_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_47AssignVariableOpAassignvariableop_47_adam_v_tcn_1_residual_block_2_conv1d_0_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_48AssignVariableOp?assignvariableop_48_adam_m_tcn_1_residual_block_2_conv1d_0_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_49AssignVariableOp?assignvariableop_49_adam_v_tcn_1_residual_block_2_conv1d_0_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_50AssignVariableOpAassignvariableop_50_adam_m_tcn_1_residual_block_2_conv1d_1_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_51AssignVariableOpAassignvariableop_51_adam_v_tcn_1_residual_block_2_conv1d_1_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_52AssignVariableOp?assignvariableop_52_adam_m_tcn_1_residual_block_2_conv1d_1_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_53AssignVariableOp?assignvariableop_53_adam_v_tcn_1_residual_block_2_conv1d_1_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_54AssignVariableOpAassignvariableop_54_adam_m_tcn_1_residual_block_3_conv1d_0_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_55AssignVariableOpAassignvariableop_55_adam_v_tcn_1_residual_block_3_conv1d_0_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_56AssignVariableOp?assignvariableop_56_adam_m_tcn_1_residual_block_3_conv1d_0_biasIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_57AssignVariableOp?assignvariableop_57_adam_v_tcn_1_residual_block_3_conv1d_0_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_58AssignVariableOpAassignvariableop_58_adam_m_tcn_1_residual_block_3_conv1d_1_kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_59AssignVariableOpAassignvariableop_59_adam_v_tcn_1_residual_block_3_conv1d_1_kernelIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_60AssignVariableOp?assignvariableop_60_adam_m_tcn_1_residual_block_3_conv1d_1_biasIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_61AssignVariableOp?assignvariableop_61_adam_v_tcn_1_residual_block_3_conv1d_1_biasIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_m_dense_1_kernelIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_63AssignVariableOp)assignvariableop_63_adam_v_dense_1_kernelIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_64AssignVariableOp'assignvariableop_64_adam_m_dense_1_biasIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_65AssignVariableOp'assignvariableop_65_adam_v_dense_1_biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_m_dense_2_kernelIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_67AssignVariableOp)assignvariableop_67_adam_v_dense_2_kernelIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_68AssignVariableOp'assignvariableop_68_adam_m_dense_2_biasIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_69AssignVariableOp'assignvariableop_69_adam_v_dense_2_biasIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_m_output_kernelIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_71AssignVariableOp(assignvariableop_71_adam_v_output_kernelIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_72AssignVariableOp&assignvariableop_72_adam_m_output_biasIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_73AssignVariableOp&assignvariableop_73_adam_v_output_biasIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_74AssignVariableOpassignvariableop_74_total_1Identity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_75AssignVariableOpassignvariableop_75_count_1Identity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_76AssignVariableOpassignvariableop_76_totalIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_77AssignVariableOpassignvariableop_77_countIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Г
Identity_78Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_79IdentityIdentity_78:output:0^NoOp_1*
T0*
_output_shapes
: р
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_79Identity_79:output:0*≥
_input_shapes°
Ю: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
∞
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15745

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15605

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15705

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
F
*__inference_SDropout_1_layer_call_fn_15610

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_12943v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15645

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_13058

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
F
*__inference_SDropout_0_layer_call_fn_15710

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_13053v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_12948

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_13053

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
F
*__inference_SDropout_0_layer_call_fn_15630

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_12965v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_12970

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_13031

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ћ
‘
A__inference_Output_layer_call_and_return_conditional_losses_13422

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ-Output/bias/Regularizer/L2Loss/ReadVariableOpҐ/Output/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€О
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ъ
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Й
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0А
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ф
Output/bias/Regularizer/mulMul&Output/bias/Regularizer/mul/x:output:0'Output/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ў
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^Output/bias/Regularizer/L2Loss/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-Output/bias/Regularizer/L2Loss/ReadVariableOp-Output/bias/Regularizer/L2Loss/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ч
F
*__inference_SDropout_1_layer_call_fn_15735

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_13080v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Г	
ђ
__inference_loss_fn_4_15576J
8output_kernel_regularizer_l2loss_readvariableop_resource: 
identityИҐ/Output/kernel/Regularizer/L2Loss/ReadVariableOp®
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp8output_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

: *
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ъ
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
∞
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_13080

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
F
*__inference_SDropout_0_layer_call_fn_15675

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_13014v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
й=
А

G__inference_sequential_1_layer_call_and_return_conditional_losses_13753
tcn_1_input!
tcn_1_13676: 
tcn_1_13678: !
tcn_1_13680:  
tcn_1_13682: !
tcn_1_13684: 
tcn_1_13686: !
tcn_1_13688:  
tcn_1_13690: !
tcn_1_13692:  
tcn_1_13694: !
tcn_1_13696:  
tcn_1_13698: !
tcn_1_13700:  
tcn_1_13702: !
tcn_1_13704:  
tcn_1_13706: !
tcn_1_13708:  
tcn_1_13710: 
dense_1_13713:  
dense_1_13715: 
dense_2_13718:  
dense_2_13720: 
output_13723: 
output_13725:
identityИҐDense_1/StatefulPartitionedCallҐ.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpҐDense_2/StatefulPartitionedCallҐ.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpҐ0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpҐOutput/StatefulPartitionedCallҐ-Output/bias/Regularizer/L2Loss/ReadVariableOpҐ/Output/kernel/Regularizer/L2Loss/ReadVariableOpҐTCN_1/StatefulPartitionedCall÷
TCN_1/StatefulPartitionedCallStatefulPartitionedCalltcn_1_inputtcn_1_13676tcn_1_13678tcn_1_13680tcn_1_13682tcn_1_13684tcn_1_13686tcn_1_13688tcn_1_13690tcn_1_13692tcn_1_13694tcn_1_13696tcn_1_13698tcn_1_13700tcn_1_13702tcn_1_13704tcn_1_13706tcn_1_13708tcn_1_13710*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_TCN_1_layer_call_and_return_conditional_losses_13675Й
Dense_1/StatefulPartitionedCallStatefulPartitionedCall&TCN_1/StatefulPartitionedCall:output:0dense_1_13713dense_1_13715*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_13372Л
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_13718dense_2_13720*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_13397З
Output/StatefulPartitionedCallStatefulPartitionedCall(Dense_2/StatefulPartitionedCall:output:0output_13723output_13725*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_Output_layer_call_and_return_conditional_losses_13422~
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_13713*
_output_shapes

:  *
dtype0Ж
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0*Dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_13715*
_output_shapes
: *
dtype0В
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0(Dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ~
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_13718*
_output_shapes

:  *
dtype0Ж
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
Dense_2/kernel/Regularizer/mulMul)Dense_2/kernel/Regularizer/mul/x:output:0*Dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_13720*
_output_shapes
: *
dtype0В
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
Dense_2/bias/Regularizer/mulMul'Dense_2/bias/Regularizer/mul/x:output:0(Dense_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_13723*
_output_shapes

: *
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ъ
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_13725*
_output_shapes
:*
dtype0А
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ф
Output/bias/Regularizer/mulMul&Output/bias/Regularizer/mul/x:output:0'Output/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€х
NoOpNoOp ^Dense_1/StatefulPartitionedCall/^Dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^Dense_2/StatefulPartitionedCall/^Dense_2/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp^Output/StatefulPartitionedCall.^Output/bias/Regularizer/L2Loss/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp^TCN_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€В: : : : : : : : : : : : : : : : : : : : : : : : 2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2`
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2`
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2^
-Output/bias/Regularizer/L2Loss/ReadVariableOp-Output/bias/Regularizer/L2Loss/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp2>
TCN_1/StatefulPartitionedCallTCN_1/StatefulPartitionedCall:Y U
,
_output_shapes
:€€€€€€€€€В
%
_user_specified_nameTCN_1_input
ђ
Ь
,__inference_sequential_1_layer_call_fn_14342

inputs
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9:  

unknown_10:  

unknown_11:  

unknown_12:  

unknown_13:  

unknown_14:  

unknown_15:  

unknown_16: 

unknown_17:  

unknown_18: 

unknown_19:  

unknown_20: 

unknown_21: 

unknown_22:
identityИҐStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_13836o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€В: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€В
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_13075

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
F
*__inference_SDropout_0_layer_call_fn_15635

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_12970v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ё
„
B__inference_Dense_1_layer_call_and_return_conditional_losses_13372

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐ.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ П
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ж
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0*Dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: К
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0В
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0(Dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ џ
NoOpNoOp^BiasAdd/ReadVariableOp/^Dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2`
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ё
„
B__inference_Dense_2_layer_call_and_return_conditional_losses_15503

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐ.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpҐ0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ П
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ж
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
Dense_2/kernel/Regularizer/mulMul)Dense_2/kernel/Regularizer/mul/x:output:0*Dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: К
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0В
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
Dense_2/bias/Regularizer/mulMul'Dense_2/bias/Regularizer/mul/x:output:0(Dense_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ џ
NoOpNoOp^BiasAdd/ReadVariableOp/^Dense_2/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2`
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
С	
Ѓ
__inference_loss_fn_0_15540K
9dense_1_kernel_regularizer_l2loss_readvariableop_resource:  
identityИҐ0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp™
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp9dense_1_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:  *
dtype0Ж
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
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
к∆
”
@__inference_TCN_1_layer_call_and_return_conditional_losses_15227

inputs[
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource: G
9residual_block_0_conv1d_0_biasadd_readvariableop_resource: [
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_0_conv1d_1_biasadd_readvariableop_resource: b
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource: N
@residual_block_0_matching_conv1d_biasadd_readvariableop_resource: [
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_1_conv1d_0_biasadd_readvariableop_resource: [
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_1_conv1d_1_biasadd_readvariableop_resource: [
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_2_conv1d_0_biasadd_readvariableop_resource: [
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_2_conv1d_1_biasadd_readvariableop_resource: [
Eresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_3_conv1d_0_biasadd_readvariableop_resource: [
Eresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_3_conv1d_1_biasadd_readvariableop_resource: 
identityИҐ0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpҐ<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpҐ<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpҐCresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpҐ<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpҐ<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpҐ<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpҐ<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpҐ<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpҐ<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpП
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ф
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Дz
/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€÷
+residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_0/Pad:output:08residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Д∆
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0s
1residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ь
 residual_block_0/conv1D_0/Conv1DConv2D4residual_block_0/conv1D_0/Conv1D/ExpandDims:output:06residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingVALID*
strides
µ
(residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_0/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€¶
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0–
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/Conv1D/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_0/Act_Conv1D_0/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_0/SDropout_0/IdentityIdentity0residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ї
residual_block_0/conv1D_1/PadPad-residual_block_0/SDropout_0/Identity:output:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Д z
/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€÷
+residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_1/Pad:output:08residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Д ∆
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ь
 residual_block_0/conv1D_1/Conv1DConv2D4residual_block_0/conv1D_1/Conv1D/ExpandDims:output:06residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingVALID*
strides
µ
(residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_1/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€¶
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0–
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/Conv1D/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_0/Act_Conv1D_1/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_0/SDropout_1/IdentityIdentity0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В У
%residual_block_0/Act_Conv_Blocks/ReluRelu-residual_block_0/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Б
6residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€ƒ
2residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputs?residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€В‘
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpLresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0z
8residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Г
4residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsKresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Aresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Р
'residual_block_0/matching_conv1D/Conv1DConv2D;residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0=residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingSAME*
strides
√
/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze0residual_block_0/matching_conv1D/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€і
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0е
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В ƒ
residual_block_0/Add_Res/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:03residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Д
#residual_block_0/Act_Res_Block/ReluRelu residual_block_0/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       њ
residual_block_1/conv1D_0/PadPad1residual_block_0/Act_Res_Block/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ж x
.residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ж®
Oresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        †
Gresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        Е
;residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ®
/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_0/Pad:output:0Dresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€C z
/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€C ∆
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_1/conv1D_0/Conv1DConv2D4residual_block_1/conv1D_0/Conv1D/ExpandDims:output:06residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€A *
paddingVALID*
strides
і
(residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€A *
squeeze_dims

э€€€€€€€€Е
;residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ±
/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_1/Act_Conv1D_0/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_1/SDropout_0/IdentityIdentity0residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ї
residual_block_1/conv1D_1/PadPad-residual_block_1/SDropout_0/Identity:output:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ж x
.residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ж®
Oresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        †
Gresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        Е
;residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ®
/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_1/Pad:output:0Dresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€C z
/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€C ∆
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_1/conv1D_1/Conv1DConv2D4residual_block_1/conv1D_1/Conv1D/ExpandDims:output:06residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€A *
paddingVALID*
strides
і
(residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€A *
squeeze_dims

э€€€€€€€€Е
;residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ±
/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_1/Act_Conv1D_1/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_1/SDropout_1/IdentityIdentity0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В У
%residual_block_1/Act_Conv_Blocks/ReluRelu-residual_block_1/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ƒ
residual_block_1/Add_Res/addAddV21residual_block_0/Act_Res_Block/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Д
#residual_block_1/Act_Res_Block/ReluRelu residual_block_1/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       њ
residual_block_2/conv1D_0/PadPad1residual_block_1/Act_Res_Block/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€К x
.residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:К®
Oresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       †
Gresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Е
;residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ®
/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_0/Pad:output:0Dresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€# z
/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€# ∆
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_2/conv1D_0/Conv1DConv2D4residual_block_2/conv1D_0/Conv1D/ExpandDims:output:06residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€! *
paddingVALID*
strides
і
(residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€! *
squeeze_dims

э€€€€€€€€Е
;residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ±
/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_2/Act_Conv1D_0/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_2/SDropout_0/IdentityIdentity0residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ї
residual_block_2/conv1D_1/PadPad-residual_block_2/SDropout_0/Identity:output:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€К x
.residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:К®
Oresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       †
Gresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Е
;residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ®
/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_1/Pad:output:0Dresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€# z
/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€# ∆
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_2/conv1D_1/Conv1DConv2D4residual_block_2/conv1D_1/Conv1D/ExpandDims:output:06residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€! *
paddingVALID*
strides
і
(residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€! *
squeeze_dims

э€€€€€€€€Е
;residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ±
/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_2/Act_Conv1D_1/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_2/SDropout_1/IdentityIdentity0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В У
%residual_block_2/Act_Conv_Blocks/ReluRelu-residual_block_2/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ƒ
residual_block_2/Add_Res/addAddV21residual_block_1/Act_Res_Block/Relu:activations:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Д
#residual_block_2/Act_Res_Block/ReluRelu residual_block_2/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       њ
residual_block_3/conv1D_0/PadPad1residual_block_2/Act_Res_Block/Relu:activations:0/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Т x
.residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Т®
Oresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       †
Gresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Е
;residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ®
/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_0/Pad:output:0Dresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€ z
/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ∆
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_3/conv1D_0/Conv1DConv2D4residual_block_3/conv1D_0/Conv1D/ExpandDims:output:06residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
і
(residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€Е
;residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ±
/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_3/conv1D_0/BiasAddBiasAdd8residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_3/Act_Conv1D_0/ReluRelu*residual_block_3/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_3/SDropout_0/IdentityIdentity0residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ї
residual_block_3/conv1D_1/PadPad-residual_block_3/SDropout_0/Identity:output:0/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Т x
.residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Т®
Oresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       †
Gresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Е
;residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ®
/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_1/Pad:output:0Dresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€ z
/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ∆
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_3/conv1D_1/Conv1DConv2D4residual_block_3/conv1D_1/Conv1D/ExpandDims:output:06residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
і
(residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€Е
;residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ±
/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_3/conv1D_1/BiasAddBiasAdd8residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_3/Act_Conv1D_1/ReluRelu*residual_block_3/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_3/SDropout_1/IdentityIdentity0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В У
%residual_block_3/Act_Conv_Blocks/ReluRelu-residual_block_3/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ƒ
residual_block_3/Add_Res/addAddV21residual_block_2/Act_Res_Block/Relu:activations:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Д
#residual_block_3/Act_Res_Block/ReluRelu residual_block_3/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В ¬
Add_Skip_Connections/addAddV23residual_block_0/Act_Conv_Blocks/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В ≠
Add_Skip_Connections/add_1AddV2Add_Skip_Connections/add:z:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В ѓ
Add_Skip_Connections/add_2AddV2Add_Skip_Connections/add_1:z:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В u
 Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    €€€€    w
"Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            w
"Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         »
Slice_Output/strided_sliceStridedSliceAdd_Skip_Connections/add_2:z:0)Slice_Output/strided_slice/stack:output:0+Slice_Output/strided_slice/stack_1:output:0+Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *

begin_mask*
end_mask*
shrink_axis_maskr
IdentityIdentity#Slice_Output/strided_slice:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ÷
NoOpNoOp1^residual_block_0/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_0/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp8^residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpD^residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€В: : : : : : : : : : : : : : : : : : 2d
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2r
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2К
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpCresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€В
 
_user_specified_nameinputs
К≠
Т
 __inference__wrapped_model_12915
tcn_1_inputn
Xsequential_1_tcn_1_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource: Z
Lsequential_1_tcn_1_residual_block_0_conv1d_0_biasadd_readvariableop_resource: n
Xsequential_1_tcn_1_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  Z
Lsequential_1_tcn_1_residual_block_0_conv1d_1_biasadd_readvariableop_resource: u
_sequential_1_tcn_1_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource: a
Ssequential_1_tcn_1_residual_block_0_matching_conv1d_biasadd_readvariableop_resource: n
Xsequential_1_tcn_1_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  Z
Lsequential_1_tcn_1_residual_block_1_conv1d_0_biasadd_readvariableop_resource: n
Xsequential_1_tcn_1_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  Z
Lsequential_1_tcn_1_residual_block_1_conv1d_1_biasadd_readvariableop_resource: n
Xsequential_1_tcn_1_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  Z
Lsequential_1_tcn_1_residual_block_2_conv1d_0_biasadd_readvariableop_resource: n
Xsequential_1_tcn_1_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  Z
Lsequential_1_tcn_1_residual_block_2_conv1d_1_biasadd_readvariableop_resource: n
Xsequential_1_tcn_1_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  Z
Lsequential_1_tcn_1_residual_block_3_conv1d_0_biasadd_readvariableop_resource: n
Xsequential_1_tcn_1_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  Z
Lsequential_1_tcn_1_residual_block_3_conv1d_1_biasadd_readvariableop_resource: E
3sequential_1_dense_1_matmul_readvariableop_resource:  B
4sequential_1_dense_1_biasadd_readvariableop_resource: E
3sequential_1_dense_2_matmul_readvariableop_resource:  B
4sequential_1_dense_2_biasadd_readvariableop_resource: D
2sequential_1_output_matmul_readvariableop_resource: A
3sequential_1_output_biasadd_readvariableop_resource:
identityИҐ+sequential_1/Dense_1/BiasAdd/ReadVariableOpҐ*sequential_1/Dense_1/MatMul/ReadVariableOpҐ+sequential_1/Dense_2/BiasAdd/ReadVariableOpҐ*sequential_1/Dense_2/MatMul/ReadVariableOpҐ*sequential_1/Output/BiasAdd/ReadVariableOpҐ)sequential_1/Output/MatMul/ReadVariableOpҐCsequential_1/TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpҐOsequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐCsequential_1/TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpҐOsequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐJsequential_1/TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpҐVsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpҐCsequential_1/TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpҐOsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐCsequential_1/TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpҐOsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐCsequential_1/TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpҐOsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐCsequential_1/TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpҐOsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐCsequential_1/TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpҐOsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐCsequential_1/TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpҐOsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ
9sequential_1/TCN_1/residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       њ
0sequential_1/TCN_1/residual_block_0/conv1D_0/PadPadtcn_1_inputBsequential_1/TCN_1/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€ДН
Bsequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€П
>sequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims9sequential_1/TCN_1/residual_block_0/conv1D_0/Pad:output:0Ksequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Дм
Osequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpXsequential_1_tcn_1_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Ж
Dsequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : І
@sequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsWsequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Msequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: µ
3sequential_1/TCN_1/residual_block_0/conv1D_0/Conv1DConv2DGsequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims:output:0Isequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingVALID*
strides
џ
;sequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze<sequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€ћ
Csequential_1/TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpLsequential_1_tcn_1_residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Й
4sequential_1/TCN_1/residual_block_0/conv1D_0/BiasAddBiasAddDsequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/Squeeze:output:0Ksequential_1/TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В ≥
5sequential_1/TCN_1/residual_block_0/Act_Conv1D_0/ReluRelu=sequential_1/TCN_1/residual_block_0/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В њ
7sequential_1/TCN_1/residual_block_0/SDropout_0/IdentityIdentityCsequential_1/TCN_1/residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Ґ
9sequential_1/TCN_1/residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ф
0sequential_1/TCN_1/residual_block_0/conv1D_1/PadPad@sequential_1/TCN_1/residual_block_0/SDropout_0/Identity:output:0Bsequential_1/TCN_1/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Д Н
Bsequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€П
>sequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims9sequential_1/TCN_1/residual_block_0/conv1D_1/Pad:output:0Ksequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Д м
Osequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpXsequential_1_tcn_1_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Ж
Dsequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : І
@sequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsWsequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Msequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  µ
3sequential_1/TCN_1/residual_block_0/conv1D_1/Conv1DConv2DGsequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims:output:0Isequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingVALID*
strides
џ
;sequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze<sequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€ћ
Csequential_1/TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpLsequential_1_tcn_1_residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Й
4sequential_1/TCN_1/residual_block_0/conv1D_1/BiasAddBiasAddDsequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/Squeeze:output:0Ksequential_1/TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В ≥
5sequential_1/TCN_1/residual_block_0/Act_Conv1D_1/ReluRelu=sequential_1/TCN_1/residual_block_0/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В њ
7sequential_1/TCN_1/residual_block_0/SDropout_1/IdentityIdentityCsequential_1/TCN_1/residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В є
8sequential_1/TCN_1/residual_block_0/Act_Conv_Blocks/ReluRelu@sequential_1/TCN_1/residual_block_0/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Ф
Isequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€п
Esequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimstcn_1_inputRsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Въ
Vsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp_sequential_1_tcn_1_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Н
Ksequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Љ
Gsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDims^sequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Tsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: …
:sequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1DConv2DNsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0Psequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingSAME*
strides
й
Bsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueezeCsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€Џ
Jsequential_1/TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOpSsequential_1_tcn_1_residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ю
;sequential_1/TCN_1/residual_block_0/matching_conv1D/BiasAddBiasAddKsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0Rsequential_1/TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В э
/sequential_1/TCN_1/residual_block_0/Add_Res/addAddV2Dsequential_1/TCN_1/residual_block_0/matching_conv1D/BiasAdd:output:0Fsequential_1/TCN_1/residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В ™
6sequential_1/TCN_1/residual_block_0/Act_Res_Block/ReluRelu3sequential_1/TCN_1/residual_block_0/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В Ґ
9sequential_1/TCN_1/residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ш
0sequential_1/TCN_1/residual_block_1/conv1D_0/PadPadDsequential_1/TCN_1/residual_block_0/Act_Res_Block/Relu:activations:0Bsequential_1/TCN_1/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ж Л
Asequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ђ
`sequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Жї
bsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ґ
]sequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ≥
Zsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        Ш
Nsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:§
Ksequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ф
Bsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND9sequential_1/TCN_1/residual_block_1/conv1D_0/Pad:output:0Wsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Tsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€C Н
Bsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€†
>sequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDimsKsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:0Ksequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€C м
Osequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpXsequential_1_tcn_1_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Ж
Dsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : І
@sequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsWsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Msequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  і
3sequential_1/TCN_1/residual_block_1/conv1D_0/Conv1DConv2DGsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims:output:0Isequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€A *
paddingVALID*
strides
Џ
;sequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze<sequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€A *
squeeze_dims

э€€€€€€€€Ш
Nsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:°
Hsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        э
Bsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceNDDsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Wsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Qsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ћ
Csequential_1/TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpLsequential_1_tcn_1_residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Р
4sequential_1/TCN_1/residual_block_1/conv1D_0/BiasAddBiasAddKsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:0Ksequential_1/TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В ≥
5sequential_1/TCN_1/residual_block_1/Act_Conv1D_0/ReluRelu=sequential_1/TCN_1/residual_block_1/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В њ
7sequential_1/TCN_1/residual_block_1/SDropout_0/IdentityIdentityCsequential_1/TCN_1/residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Ґ
9sequential_1/TCN_1/residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ф
0sequential_1/TCN_1/residual_block_1/conv1D_1/PadPad@sequential_1/TCN_1/residual_block_1/SDropout_0/Identity:output:0Bsequential_1/TCN_1/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ж Л
Asequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ђ
`sequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Жї
bsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ґ
]sequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ≥
Zsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        Ш
Nsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:§
Ksequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ф
Bsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND9sequential_1/TCN_1/residual_block_1/conv1D_1/Pad:output:0Wsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Tsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€C Н
Bsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€†
>sequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDimsKsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:0Ksequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€C м
Osequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpXsequential_1_tcn_1_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Ж
Dsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : І
@sequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsWsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Msequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  і
3sequential_1/TCN_1/residual_block_1/conv1D_1/Conv1DConv2DGsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims:output:0Isequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€A *
paddingVALID*
strides
Џ
;sequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze<sequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€A *
squeeze_dims

э€€€€€€€€Ш
Nsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:°
Hsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        э
Bsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceNDDsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Wsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Qsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ћ
Csequential_1/TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpLsequential_1_tcn_1_residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Р
4sequential_1/TCN_1/residual_block_1/conv1D_1/BiasAddBiasAddKsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:0Ksequential_1/TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В ≥
5sequential_1/TCN_1/residual_block_1/Act_Conv1D_1/ReluRelu=sequential_1/TCN_1/residual_block_1/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В њ
7sequential_1/TCN_1/residual_block_1/SDropout_1/IdentityIdentityCsequential_1/TCN_1/residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В є
8sequential_1/TCN_1/residual_block_1/Act_Conv_Blocks/ReluRelu@sequential_1/TCN_1/residual_block_1/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В э
/sequential_1/TCN_1/residual_block_1/Add_Res/addAddV2Dsequential_1/TCN_1/residual_block_0/Act_Res_Block/Relu:activations:0Fsequential_1/TCN_1/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В ™
6sequential_1/TCN_1/residual_block_1/Act_Res_Block/ReluRelu3sequential_1/TCN_1/residual_block_1/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В Ґ
9sequential_1/TCN_1/residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ш
0sequential_1/TCN_1/residual_block_2/conv1D_0/PadPadDsequential_1/TCN_1/residual_block_1/Act_Res_Block/Relu:activations:0Bsequential_1/TCN_1/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€К Л
Asequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ђ
`sequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Кї
bsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ґ
]sequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ≥
Zsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Ш
Nsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:§
Ksequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ф
Bsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND9sequential_1/TCN_1/residual_block_2/conv1D_0/Pad:output:0Wsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Tsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€# Н
Bsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€†
>sequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDimsKsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:0Ksequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€# м
Osequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpXsequential_1_tcn_1_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Ж
Dsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : І
@sequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsWsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Msequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  і
3sequential_1/TCN_1/residual_block_2/conv1D_0/Conv1DConv2DGsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims:output:0Isequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€! *
paddingVALID*
strides
Џ
;sequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze<sequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€! *
squeeze_dims

э€€€€€€€€Ш
Nsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:°
Hsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       э
Bsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceNDDsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Wsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Qsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ћ
Csequential_1/TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpLsequential_1_tcn_1_residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Р
4sequential_1/TCN_1/residual_block_2/conv1D_0/BiasAddBiasAddKsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:0Ksequential_1/TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В ≥
5sequential_1/TCN_1/residual_block_2/Act_Conv1D_0/ReluRelu=sequential_1/TCN_1/residual_block_2/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В њ
7sequential_1/TCN_1/residual_block_2/SDropout_0/IdentityIdentityCsequential_1/TCN_1/residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Ґ
9sequential_1/TCN_1/residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ф
0sequential_1/TCN_1/residual_block_2/conv1D_1/PadPad@sequential_1/TCN_1/residual_block_2/SDropout_0/Identity:output:0Bsequential_1/TCN_1/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€К Л
Asequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ђ
`sequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Кї
bsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ґ
]sequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ≥
Zsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Ш
Nsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:§
Ksequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ф
Bsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND9sequential_1/TCN_1/residual_block_2/conv1D_1/Pad:output:0Wsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Tsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€# Н
Bsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€†
>sequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDimsKsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:0Ksequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€# м
Osequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpXsequential_1_tcn_1_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Ж
Dsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : І
@sequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsWsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Msequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  і
3sequential_1/TCN_1/residual_block_2/conv1D_1/Conv1DConv2DGsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims:output:0Isequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€! *
paddingVALID*
strides
Џ
;sequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze<sequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€! *
squeeze_dims

э€€€€€€€€Ш
Nsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:°
Hsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       э
Bsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceNDDsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Wsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Qsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ћ
Csequential_1/TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpLsequential_1_tcn_1_residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Р
4sequential_1/TCN_1/residual_block_2/conv1D_1/BiasAddBiasAddKsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:0Ksequential_1/TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В ≥
5sequential_1/TCN_1/residual_block_2/Act_Conv1D_1/ReluRelu=sequential_1/TCN_1/residual_block_2/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В њ
7sequential_1/TCN_1/residual_block_2/SDropout_1/IdentityIdentityCsequential_1/TCN_1/residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В є
8sequential_1/TCN_1/residual_block_2/Act_Conv_Blocks/ReluRelu@sequential_1/TCN_1/residual_block_2/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В э
/sequential_1/TCN_1/residual_block_2/Add_Res/addAddV2Dsequential_1/TCN_1/residual_block_1/Act_Res_Block/Relu:activations:0Fsequential_1/TCN_1/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В ™
6sequential_1/TCN_1/residual_block_2/Act_Res_Block/ReluRelu3sequential_1/TCN_1/residual_block_2/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В Ґ
9sequential_1/TCN_1/residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ш
0sequential_1/TCN_1/residual_block_3/conv1D_0/PadPadDsequential_1/TCN_1/residual_block_2/Act_Res_Block/Relu:activations:0Bsequential_1/TCN_1/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Т Л
Asequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ђ
`sequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Тї
bsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ґ
]sequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ≥
Zsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Ш
Nsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:§
Ksequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ф
Bsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND9sequential_1/TCN_1/residual_block_3/conv1D_0/Pad:output:0Wsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Tsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Н
Bsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€†
>sequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDimsKsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:0Ksequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ м
Osequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpXsequential_1_tcn_1_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Ж
Dsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : І
@sequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsWsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0Msequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  і
3sequential_1/TCN_1/residual_block_3/conv1D_0/Conv1DConv2DGsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims:output:0Isequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
Џ
;sequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze<sequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€Ш
Nsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:°
Hsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       э
Bsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceNDDsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Wsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0Qsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ћ
Csequential_1/TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpLsequential_1_tcn_1_residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Р
4sequential_1/TCN_1/residual_block_3/conv1D_0/BiasAddBiasAddKsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:0Ksequential_1/TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В ≥
5sequential_1/TCN_1/residual_block_3/Act_Conv1D_0/ReluRelu=sequential_1/TCN_1/residual_block_3/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В њ
7sequential_1/TCN_1/residual_block_3/SDropout_0/IdentityIdentityCsequential_1/TCN_1/residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Ґ
9sequential_1/TCN_1/residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ф
0sequential_1/TCN_1/residual_block_3/conv1D_1/PadPad@sequential_1/TCN_1/residual_block_3/SDropout_0/Identity:output:0Bsequential_1/TCN_1/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Т Л
Asequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ђ
`sequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Тї
bsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ґ
]sequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ≥
Zsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Ш
Nsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:§
Ksequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ф
Bsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND9sequential_1/TCN_1/residual_block_3/conv1D_1/Pad:output:0Wsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Tsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Н
Bsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€†
>sequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDimsKsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:0Ksequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ м
Osequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpXsequential_1_tcn_1_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Ж
Dsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : І
@sequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsWsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Msequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  і
3sequential_1/TCN_1/residual_block_3/conv1D_1/Conv1DConv2DGsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims:output:0Isequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
Џ
;sequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze<sequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€Ш
Nsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:°
Hsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       э
Bsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceNDDsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Wsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0Qsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ћ
Csequential_1/TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpLsequential_1_tcn_1_residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Р
4sequential_1/TCN_1/residual_block_3/conv1D_1/BiasAddBiasAddKsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:0Ksequential_1/TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В ≥
5sequential_1/TCN_1/residual_block_3/Act_Conv1D_1/ReluRelu=sequential_1/TCN_1/residual_block_3/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В њ
7sequential_1/TCN_1/residual_block_3/SDropout_1/IdentityIdentityCsequential_1/TCN_1/residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В є
8sequential_1/TCN_1/residual_block_3/Act_Conv_Blocks/ReluRelu@sequential_1/TCN_1/residual_block_3/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В э
/sequential_1/TCN_1/residual_block_3/Add_Res/addAddV2Dsequential_1/TCN_1/residual_block_2/Act_Res_Block/Relu:activations:0Fsequential_1/TCN_1/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В ™
6sequential_1/TCN_1/residual_block_3/Act_Res_Block/ReluRelu3sequential_1/TCN_1/residual_block_3/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В ы
+sequential_1/TCN_1/Add_Skip_Connections/addAddV2Fsequential_1/TCN_1/residual_block_0/Act_Conv_Blocks/Relu:activations:0Fsequential_1/TCN_1/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В ж
-sequential_1/TCN_1/Add_Skip_Connections/add_1AddV2/sequential_1/TCN_1/Add_Skip_Connections/add:z:0Fsequential_1/TCN_1/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В и
-sequential_1/TCN_1/Add_Skip_Connections/add_2AddV21sequential_1/TCN_1/Add_Skip_Connections/add_1:z:0Fsequential_1/TCN_1/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В И
3sequential_1/TCN_1/Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    €€€€    К
5sequential_1/TCN_1/Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            К
5sequential_1/TCN_1/Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         І
-sequential_1/TCN_1/Slice_Output/strided_sliceStridedSlice1sequential_1/TCN_1/Add_Skip_Connections/add_2:z:0<sequential_1/TCN_1/Slice_Output/strided_slice/stack:output:0>sequential_1/TCN_1/Slice_Output/strided_slice/stack_1:output:0>sequential_1/TCN_1/Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *

begin_mask*
end_mask*
shrink_axis_maskЮ
*sequential_1/Dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0√
sequential_1/Dense_1/MatMulMatMul6sequential_1/TCN_1/Slice_Output/strided_slice:output:02sequential_1/Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ь
+sequential_1/Dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0µ
sequential_1/Dense_1/BiasAddBiasAdd%sequential_1/Dense_1/MatMul:product:03sequential_1/Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ z
sequential_1/Dense_1/ReluRelu%sequential_1/Dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ю
*sequential_1/Dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0і
sequential_1/Dense_2/MatMulMatMul'sequential_1/Dense_1/Relu:activations:02sequential_1/Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ь
+sequential_1/Dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0µ
sequential_1/Dense_2/BiasAddBiasAdd%sequential_1/Dense_2/MatMul:product:03sequential_1/Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ z
sequential_1/Dense_2/ReluRelu%sequential_1/Dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ь
)sequential_1/Output/MatMul/ReadVariableOpReadVariableOp2sequential_1_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype0≤
sequential_1/Output/MatMulMatMul'sequential_1/Dense_2/Relu:activations:01sequential_1/Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
*sequential_1/Output/BiasAdd/ReadVariableOpReadVariableOp3sequential_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0≤
sequential_1/Output/BiasAddBiasAdd$sequential_1/Output/MatMul:product:02sequential_1/Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€~
sequential_1/Output/SoftmaxSoftmax$sequential_1/Output/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€t
IdentityIdentity%sequential_1/Output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ї
NoOpNoOp,^sequential_1/Dense_1/BiasAdd/ReadVariableOp+^sequential_1/Dense_1/MatMul/ReadVariableOp,^sequential_1/Dense_2/BiasAdd/ReadVariableOp+^sequential_1/Dense_2/MatMul/ReadVariableOp+^sequential_1/Output/BiasAdd/ReadVariableOp*^sequential_1/Output/MatMul/ReadVariableOpD^sequential_1/TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpP^sequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpD^sequential_1/TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpP^sequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpK^sequential_1/TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpW^sequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpD^sequential_1/TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpP^sequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpD^sequential_1/TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpP^sequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpD^sequential_1/TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpP^sequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpD^sequential_1/TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpP^sequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpD^sequential_1/TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpP^sequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpD^sequential_1/TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpP^sequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€В: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+sequential_1/Dense_1/BiasAdd/ReadVariableOp+sequential_1/Dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/Dense_1/MatMul/ReadVariableOp*sequential_1/Dense_1/MatMul/ReadVariableOp2Z
+sequential_1/Dense_2/BiasAdd/ReadVariableOp+sequential_1/Dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/Dense_2/MatMul/ReadVariableOp*sequential_1/Dense_2/MatMul/ReadVariableOp2X
*sequential_1/Output/BiasAdd/ReadVariableOp*sequential_1/Output/BiasAdd/ReadVariableOp2V
)sequential_1/Output/MatMul/ReadVariableOp)sequential_1/Output/MatMul/ReadVariableOp2К
Csequential_1/TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpCsequential_1/TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2Ґ
Osequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpOsequential_1/TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2К
Csequential_1/TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpCsequential_1/TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2Ґ
Osequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpOsequential_1/TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2Ш
Jsequential_1/TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpJsequential_1/TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2∞
Vsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpVsequential_1/TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2К
Csequential_1/TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpCsequential_1/TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2Ґ
Osequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpOsequential_1/TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2К
Csequential_1/TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpCsequential_1/TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2Ґ
Osequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpOsequential_1/TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2К
Csequential_1/TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpCsequential_1/TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2Ґ
Osequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpOsequential_1/TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2К
Csequential_1/TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpCsequential_1/TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2Ґ
Osequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpOsequential_1/TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2К
Csequential_1/TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpCsequential_1/TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2Ґ
Osequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpOsequential_1/TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2К
Csequential_1/TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpCsequential_1/TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2Ґ
Osequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpOsequential_1/TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:Y U
,
_output_shapes
:€€€€€€€€€В
%
_user_specified_nameTCN_1_input
∞
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_12965

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ё
„
B__inference_Dense_1_layer_call_and_return_conditional_losses_15475

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐ.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ П
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ж
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0*Dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: К
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0В
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0(Dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ џ
NoOpNoOp^BiasAdd/ReadVariableOp/^Dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2`
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ђ
Ь
,__inference_sequential_1_layer_call_fn_14395

inputs
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9:  

unknown_10:  

unknown_11:  

unknown_12:  

unknown_13:  

unknown_14:  

unknown_15:  

unknown_16: 

unknown_17:  

unknown_18: 

unknown_19:  

unknown_20: 

unknown_21: 

unknown_22:
identityИҐStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_13969o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€В: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€В
 
_user_specified_nameinputs
н
¶
__inference_loss_fn_1_15549E
7dense_1_bias_regularizer_l2loss_readvariableop_resource: 
identityИҐ.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp7dense_1_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
: *
dtype0В
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
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
∞
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_12987

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
√
О
#__inference_signature_wrapper_12673
x
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9:  

unknown_10:  

unknown_11:  

unknown_12:  

unknown_13:  

unknown_14:  

unknown_15:  

unknown_16: 

unknown_17:  

unknown_18: 

unknown_19:  

unknown_20: 

unknown_21: 

unknown_22:
identityИҐStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *#
fR
__inference_<lambda>_12618f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:В: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:В

_user_specified_namex
∞
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15720

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
F
*__inference_SDropout_0_layer_call_fn_15670

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_13009v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
™Ш
о
G__inference_sequential_1_layer_call_and_return_conditional_losses_14925

inputsa
Ktcn_1_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource: M
?tcn_1_residual_block_0_conv1d_0_biasadd_readvariableop_resource: a
Ktcn_1_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  M
?tcn_1_residual_block_0_conv1d_1_biasadd_readvariableop_resource: h
Rtcn_1_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource: T
Ftcn_1_residual_block_0_matching_conv1d_biasadd_readvariableop_resource: a
Ktcn_1_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  M
?tcn_1_residual_block_1_conv1d_0_biasadd_readvariableop_resource: a
Ktcn_1_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  M
?tcn_1_residual_block_1_conv1d_1_biasadd_readvariableop_resource: a
Ktcn_1_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  M
?tcn_1_residual_block_2_conv1d_0_biasadd_readvariableop_resource: a
Ktcn_1_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  M
?tcn_1_residual_block_2_conv1d_1_biasadd_readvariableop_resource: a
Ktcn_1_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  M
?tcn_1_residual_block_3_conv1d_0_biasadd_readvariableop_resource: a
Ktcn_1_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  M
?tcn_1_residual_block_3_conv1d_1_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:  5
'dense_1_biasadd_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource:  5
'dense_2_biasadd_readvariableop_resource: 7
%output_matmul_readvariableop_resource: 4
&output_biasadd_readvariableop_resource:
identityИҐDense_1/BiasAdd/ReadVariableOpҐDense_1/MatMul/ReadVariableOpҐ.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpҐDense_2/BiasAdd/ReadVariableOpҐDense_2/MatMul/ReadVariableOpҐ.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpҐ0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpҐOutput/BiasAdd/ReadVariableOpҐOutput/MatMul/ReadVariableOpҐ-Output/bias/Regularizer/L2Loss/ReadVariableOpҐ/Output/kernel/Regularizer/L2Loss/ReadVariableOpҐ6TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpҐBTCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ6TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpҐBTCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ=TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpҐITCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpҐ6TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpҐBTCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ6TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpҐBTCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ6TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpҐBTCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ6TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpҐBTCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ6TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpҐBTCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ6TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpҐBTCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpХ
,TCN_1/residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       †
#TCN_1/residual_block_0/conv1D_0/PadPadinputs5TCN_1/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€ДА
5TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€и
1TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims,TCN_1/residual_block_0/conv1D_0/Pad:output:0>TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Д“
BTCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpKtcn_1_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0y
7TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
3TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsJTCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0@TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: О
&TCN_1/residual_block_0/conv1D_0/Conv1DConv2D:TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims:output:0<TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingVALID*
strides
Ѕ
.TCN_1/residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze/TCN_1/residual_block_0/conv1D_0/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€≤
6TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp?tcn_1_residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0в
'TCN_1/residual_block_0/conv1D_0/BiasAddBiasAdd7TCN_1/residual_block_0/conv1D_0/Conv1D/Squeeze:output:0>TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
(TCN_1/residual_block_0/Act_Conv1D_0/ReluRelu0TCN_1/residual_block_0/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В •
*TCN_1/residual_block_0/SDropout_0/IdentityIdentity6TCN_1/residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Х
,TCN_1/residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ќ
#TCN_1/residual_block_0/conv1D_1/PadPad3TCN_1/residual_block_0/SDropout_0/Identity:output:05TCN_1/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Д А
5TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€и
1TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims,TCN_1/residual_block_0/conv1D_1/Pad:output:0>TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Д “
BTCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpKtcn_1_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0y
7TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
3TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsJTCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0@TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  О
&TCN_1/residual_block_0/conv1D_1/Conv1DConv2D:TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims:output:0<TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingVALID*
strides
Ѕ
.TCN_1/residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze/TCN_1/residual_block_0/conv1D_1/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€≤
6TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp?tcn_1_residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0в
'TCN_1/residual_block_0/conv1D_1/BiasAddBiasAdd7TCN_1/residual_block_0/conv1D_1/Conv1D/Squeeze:output:0>TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
(TCN_1/residual_block_0/Act_Conv1D_1/ReluRelu0TCN_1/residual_block_0/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В •
*TCN_1/residual_block_0/SDropout_1/IdentityIdentity6TCN_1/residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Я
+TCN_1/residual_block_0/Act_Conv_Blocks/ReluRelu3TCN_1/residual_block_0/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В З
<TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€–
8TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputsETCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ва
ITCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpRtcn_1_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0А
>TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Х
:TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsQTCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0GTCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Ґ
-TCN_1/residual_block_0/matching_conv1D/Conv1DConv2DATCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0CTCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingSAME*
strides
ѕ
5TCN_1/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze6TCN_1/residual_block_0/matching_conv1D/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€ј
=TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOpFtcn_1_residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ч
.TCN_1/residual_block_0/matching_conv1D/BiasAddBiasAdd>TCN_1/residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0ETCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В ÷
"TCN_1/residual_block_0/Add_Res/addAddV27TCN_1/residual_block_0/matching_conv1D/BiasAdd:output:09TCN_1/residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Р
)TCN_1/residual_block_0/Act_Res_Block/ReluRelu&TCN_1/residual_block_0/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В Х
,TCN_1/residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       —
#TCN_1/residual_block_1/conv1D_0/PadPad7TCN_1/residual_block_0/Act_Res_Block/Relu:activations:05TCN_1/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ж ~
4TCN_1/residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ю
STCN_1/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ЖЃ
UTCN_1/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ©
PTCN_1/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¶
MTCN_1/residual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        Л
ATCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ч
>TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ј
5TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND,TCN_1/residual_block_1/conv1D_0/Pad:output:0JTCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0GTCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€C А
5TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€щ
1TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims>TCN_1/residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:0>TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€C “
BTCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpKtcn_1_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0y
7TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
3TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsJTCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0@TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Н
&TCN_1/residual_block_1/conv1D_0/Conv1DConv2D:TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims:output:0<TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€A *
paddingVALID*
strides
ј
.TCN_1/residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze/TCN_1/residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€A *
squeeze_dims

э€€€€€€€€Л
ATCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ф
;TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        …
5TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND7TCN_1/residual_block_1/conv1D_0/Conv1D/Squeeze:output:0JTCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0DTCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ≤
6TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp?tcn_1_residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0й
'TCN_1/residual_block_1/conv1D_0/BiasAddBiasAdd>TCN_1/residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:0>TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
(TCN_1/residual_block_1/Act_Conv1D_0/ReluRelu0TCN_1/residual_block_1/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В •
*TCN_1/residual_block_1/SDropout_0/IdentityIdentity6TCN_1/residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Х
,TCN_1/residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ќ
#TCN_1/residual_block_1/conv1D_1/PadPad3TCN_1/residual_block_1/SDropout_0/Identity:output:05TCN_1/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ж ~
4TCN_1/residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ю
STCN_1/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ЖЃ
UTCN_1/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ©
PTCN_1/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ¶
MTCN_1/residual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        Л
ATCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ч
>TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ј
5TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND,TCN_1/residual_block_1/conv1D_1/Pad:output:0JTCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0GTCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€C А
5TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€щ
1TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims>TCN_1/residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:0>TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€C “
BTCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpKtcn_1_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0y
7TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
3TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsJTCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0@TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Н
&TCN_1/residual_block_1/conv1D_1/Conv1DConv2D:TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims:output:0<TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€A *
paddingVALID*
strides
ј
.TCN_1/residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze/TCN_1/residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€A *
squeeze_dims

э€€€€€€€€Л
ATCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ф
;TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        …
5TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND7TCN_1/residual_block_1/conv1D_1/Conv1D/Squeeze:output:0JTCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0DTCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ≤
6TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp?tcn_1_residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0й
'TCN_1/residual_block_1/conv1D_1/BiasAddBiasAdd>TCN_1/residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:0>TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
(TCN_1/residual_block_1/Act_Conv1D_1/ReluRelu0TCN_1/residual_block_1/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В •
*TCN_1/residual_block_1/SDropout_1/IdentityIdentity6TCN_1/residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Я
+TCN_1/residual_block_1/Act_Conv_Blocks/ReluRelu3TCN_1/residual_block_1/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ÷
"TCN_1/residual_block_1/Add_Res/addAddV27TCN_1/residual_block_0/Act_Res_Block/Relu:activations:09TCN_1/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Р
)TCN_1/residual_block_1/Act_Res_Block/ReluRelu&TCN_1/residual_block_1/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В Х
,TCN_1/residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       —
#TCN_1/residual_block_2/conv1D_0/PadPad7TCN_1/residual_block_1/Act_Res_Block/Relu:activations:05TCN_1/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€К ~
4TCN_1/residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ю
STCN_1/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:КЃ
UTCN_1/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ©
PTCN_1/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¶
MTCN_1/residual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Л
ATCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ч
>TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ј
5TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND,TCN_1/residual_block_2/conv1D_0/Pad:output:0JTCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0GTCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€# А
5TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€щ
1TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims>TCN_1/residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:0>TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€# “
BTCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpKtcn_1_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0y
7TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
3TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsJTCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0@TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Н
&TCN_1/residual_block_2/conv1D_0/Conv1DConv2D:TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims:output:0<TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€! *
paddingVALID*
strides
ј
.TCN_1/residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze/TCN_1/residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€! *
squeeze_dims

э€€€€€€€€Л
ATCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ф
;TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       …
5TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND7TCN_1/residual_block_2/conv1D_0/Conv1D/Squeeze:output:0JTCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0DTCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ≤
6TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp?tcn_1_residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0й
'TCN_1/residual_block_2/conv1D_0/BiasAddBiasAdd>TCN_1/residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:0>TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
(TCN_1/residual_block_2/Act_Conv1D_0/ReluRelu0TCN_1/residual_block_2/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В •
*TCN_1/residual_block_2/SDropout_0/IdentityIdentity6TCN_1/residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Х
,TCN_1/residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ќ
#TCN_1/residual_block_2/conv1D_1/PadPad3TCN_1/residual_block_2/SDropout_0/Identity:output:05TCN_1/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€К ~
4TCN_1/residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ю
STCN_1/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:КЃ
UTCN_1/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ©
PTCN_1/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¶
MTCN_1/residual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Л
ATCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ч
>TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ј
5TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND,TCN_1/residual_block_2/conv1D_1/Pad:output:0JTCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0GTCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€# А
5TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€щ
1TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims>TCN_1/residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:0>TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€# “
BTCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpKtcn_1_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0y
7TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
3TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsJTCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0@TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Н
&TCN_1/residual_block_2/conv1D_1/Conv1DConv2D:TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims:output:0<TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€! *
paddingVALID*
strides
ј
.TCN_1/residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze/TCN_1/residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€! *
squeeze_dims

э€€€€€€€€Л
ATCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ф
;TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       …
5TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND7TCN_1/residual_block_2/conv1D_1/Conv1D/Squeeze:output:0JTCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0DTCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ≤
6TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp?tcn_1_residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0й
'TCN_1/residual_block_2/conv1D_1/BiasAddBiasAdd>TCN_1/residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:0>TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
(TCN_1/residual_block_2/Act_Conv1D_1/ReluRelu0TCN_1/residual_block_2/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В •
*TCN_1/residual_block_2/SDropout_1/IdentityIdentity6TCN_1/residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Я
+TCN_1/residual_block_2/Act_Conv_Blocks/ReluRelu3TCN_1/residual_block_2/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ÷
"TCN_1/residual_block_2/Add_Res/addAddV27TCN_1/residual_block_1/Act_Res_Block/Relu:activations:09TCN_1/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Р
)TCN_1/residual_block_2/Act_Res_Block/ReluRelu&TCN_1/residual_block_2/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В Х
,TCN_1/residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       —
#TCN_1/residual_block_3/conv1D_0/PadPad7TCN_1/residual_block_2/Act_Res_Block/Relu:activations:05TCN_1/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Т ~
4TCN_1/residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ю
STCN_1/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ТЃ
UTCN_1/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ©
PTCN_1/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¶
MTCN_1/residual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Л
ATCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ч
>TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ј
5TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND,TCN_1/residual_block_3/conv1D_0/Pad:output:0JTCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0GTCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€ А
5TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€щ
1TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims>TCN_1/residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:0>TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ “
BTCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpKtcn_1_residual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0y
7TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
3TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsJTCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0@TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Н
&TCN_1/residual_block_3/conv1D_0/Conv1DConv2D:TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims:output:0<TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
ј
.TCN_1/residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze/TCN_1/residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€Л
ATCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ф
;TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       …
5TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND7TCN_1/residual_block_3/conv1D_0/Conv1D/Squeeze:output:0JTCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0DTCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ≤
6TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp?tcn_1_residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0й
'TCN_1/residual_block_3/conv1D_0/BiasAddBiasAdd>TCN_1/residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:0>TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
(TCN_1/residual_block_3/Act_Conv1D_0/ReluRelu0TCN_1/residual_block_3/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В •
*TCN_1/residual_block_3/SDropout_0/IdentityIdentity6TCN_1/residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Х
,TCN_1/residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ќ
#TCN_1/residual_block_3/conv1D_1/PadPad3TCN_1/residual_block_3/SDropout_0/Identity:output:05TCN_1/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Т ~
4TCN_1/residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ю
STCN_1/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ТЃ
UTCN_1/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ©
PTCN_1/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ¶
MTCN_1/residual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Л
ATCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ч
>TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ј
5TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND,TCN_1/residual_block_3/conv1D_1/Pad:output:0JTCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0GTCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€ А
5TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€щ
1TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims>TCN_1/residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:0>TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ “
BTCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpKtcn_1_residual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0y
7TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : А
3TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsJTCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0@TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  Н
&TCN_1/residual_block_3/conv1D_1/Conv1DConv2D:TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims:output:0<TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
ј
.TCN_1/residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze/TCN_1/residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€Л
ATCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ф
;TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       …
5TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND7TCN_1/residual_block_3/conv1D_1/Conv1D/Squeeze:output:0JTCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0DTCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ≤
6TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp?tcn_1_residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0й
'TCN_1/residual_block_3/conv1D_1/BiasAddBiasAdd>TCN_1/residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:0>TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
(TCN_1/residual_block_3/Act_Conv1D_1/ReluRelu0TCN_1/residual_block_3/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В •
*TCN_1/residual_block_3/SDropout_1/IdentityIdentity6TCN_1/residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Я
+TCN_1/residual_block_3/Act_Conv_Blocks/ReluRelu3TCN_1/residual_block_3/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ÷
"TCN_1/residual_block_3/Add_Res/addAddV27TCN_1/residual_block_2/Act_Res_Block/Relu:activations:09TCN_1/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Р
)TCN_1/residual_block_3/Act_Res_Block/ReluRelu&TCN_1/residual_block_3/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В ‘
TCN_1/Add_Skip_Connections/addAddV29TCN_1/residual_block_0/Act_Conv_Blocks/Relu:activations:09TCN_1/residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В њ
 TCN_1/Add_Skip_Connections/add_1AddV2"TCN_1/Add_Skip_Connections/add:z:09TCN_1/residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Ѕ
 TCN_1/Add_Skip_Connections/add_2AddV2$TCN_1/Add_Skip_Connections/add_1:z:09TCN_1/residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В {
&TCN_1/Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    €€€€    }
(TCN_1/Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            }
(TCN_1/Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ж
 TCN_1/Slice_Output/strided_sliceStridedSlice$TCN_1/Add_Skip_Connections/add_2:z:0/TCN_1/Slice_Output/strided_slice/stack:output:01TCN_1/Slice_Output/strided_slice/stack_1:output:01TCN_1/Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *

begin_mask*
end_mask*
shrink_axis_maskД
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ь
Dense_1/MatMulMatMul)TCN_1/Slice_Output/strided_slice:output:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ В
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ `
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Д
Dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Н
Dense_2/MatMulMatMulDense_1/Relu:activations:0%Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ В
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
Dense_2/BiasAddBiasAddDense_2/MatMul:product:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ `
Dense_2/ReluReluDense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ В
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Л
Output/MatMulMatMulDense_2/Relu:activations:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€А
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ч
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ж
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0*Dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Т
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0В
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0(Dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ч
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ж
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
Dense_2/kernel/Regularizer/mulMul)Dense_2/kernel/Regularizer/mul/x:output:0*Dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Т
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0В
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
Dense_2/bias/Regularizer/mulMul'Dense_2/bias/Regularizer/mul/x:output:0(Dense_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Х
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ъ
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Р
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0А
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ф
Output/bias/Regularizer/mulMul&Output/bias/Regularizer/mul/x:output:0'Output/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityOutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≠
NoOpNoOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp/^Dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp^Dense_2/MatMul/ReadVariableOp/^Dense_2/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp.^Output/bias/Regularizer/L2Loss/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp7^TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpC^TCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp7^TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpC^TCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp>^TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpJ^TCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp7^TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpC^TCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp7^TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpC^TCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp7^TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpC^TCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp7^TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpC^TCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp7^TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOpC^TCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp7^TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOpC^TCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€В: : : : : : : : : : : : : : : : : : : : : : : : 2@
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
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp2p
6TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp6TCN_1/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2И
BTCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpBTCN_1/residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2p
6TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp6TCN_1/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2И
BTCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpBTCN_1/residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2~
=TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp=TCN_1/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2Ц
ITCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpITCN_1/residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2p
6TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp6TCN_1/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2И
BTCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpBTCN_1/residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2p
6TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp6TCN_1/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2И
BTCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpBTCN_1/residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2p
6TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp6TCN_1/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2И
BTCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpBTCN_1/residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2p
6TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp6TCN_1/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2И
BTCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpBTCN_1/residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2p
6TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp6TCN_1/residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2И
BTCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpBTCN_1/residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2p
6TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp6TCN_1/residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2И
BTCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpBTCN_1/residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€В
 
_user_specified_nameinputs
С	
Ѓ
__inference_loss_fn_2_15558K
9dense_2_kernel_regularizer_l2loss_readvariableop_resource:  
identityИҐ0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp™
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp9dense_2_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:  *
dtype0Ж
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
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
ч
F
*__inference_SDropout_1_layer_call_fn_15690

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_13031v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15725

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ї
°
,__inference_sequential_1_layer_call_fn_14020
tcn_1_input
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9:  

unknown_10:  

unknown_11:  

unknown_12:  

unknown_13:  

unknown_14:  

unknown_15:  

unknown_16: 

unknown_17:  

unknown_18: 

unknown_19:  

unknown_20: 

unknown_21: 

unknown_22:
identityИҐStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCalltcn_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_13969o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€В: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:€€€€€€€€€В
%
_user_specified_nameTCN_1_input
ї¬
щP
__inference__traced_save_16236
file_prefix7
%read_disablecopyonread_dense_1_kernel:  3
%read_1_disablecopyonread_dense_1_bias: 9
'read_2_disablecopyonread_dense_2_kernel:  3
%read_3_disablecopyonread_dense_2_bias: 8
&read_4_disablecopyonread_output_kernel: 2
$read_5_disablecopyonread_output_bias:U
?read_6_disablecopyonread_tcn_1_residual_block_0_conv1d_0_kernel: K
=read_7_disablecopyonread_tcn_1_residual_block_0_conv1d_0_bias: U
?read_8_disablecopyonread_tcn_1_residual_block_0_conv1d_1_kernel:  K
=read_9_disablecopyonread_tcn_1_residual_block_0_conv1d_1_bias: ]
Gread_10_disablecopyonread_tcn_1_residual_block_0_matching_conv1d_kernel: S
Eread_11_disablecopyonread_tcn_1_residual_block_0_matching_conv1d_bias: V
@read_12_disablecopyonread_tcn_1_residual_block_1_conv1d_0_kernel:  L
>read_13_disablecopyonread_tcn_1_residual_block_1_conv1d_0_bias: V
@read_14_disablecopyonread_tcn_1_residual_block_1_conv1d_1_kernel:  L
>read_15_disablecopyonread_tcn_1_residual_block_1_conv1d_1_bias: V
@read_16_disablecopyonread_tcn_1_residual_block_2_conv1d_0_kernel:  L
>read_17_disablecopyonread_tcn_1_residual_block_2_conv1d_0_bias: V
@read_18_disablecopyonread_tcn_1_residual_block_2_conv1d_1_kernel:  L
>read_19_disablecopyonread_tcn_1_residual_block_2_conv1d_1_bias: V
@read_20_disablecopyonread_tcn_1_residual_block_3_conv1d_0_kernel:  L
>read_21_disablecopyonread_tcn_1_residual_block_3_conv1d_0_bias: V
@read_22_disablecopyonread_tcn_1_residual_block_3_conv1d_1_kernel:  L
>read_23_disablecopyonread_tcn_1_residual_block_3_conv1d_1_bias: -
#read_24_disablecopyonread_iteration:	 1
'read_25_disablecopyonread_learning_rate: ]
Gread_26_disablecopyonread_adam_m_tcn_1_residual_block_0_conv1d_0_kernel: ]
Gread_27_disablecopyonread_adam_v_tcn_1_residual_block_0_conv1d_0_kernel: S
Eread_28_disablecopyonread_adam_m_tcn_1_residual_block_0_conv1d_0_bias: S
Eread_29_disablecopyonread_adam_v_tcn_1_residual_block_0_conv1d_0_bias: ]
Gread_30_disablecopyonread_adam_m_tcn_1_residual_block_0_conv1d_1_kernel:  ]
Gread_31_disablecopyonread_adam_v_tcn_1_residual_block_0_conv1d_1_kernel:  S
Eread_32_disablecopyonread_adam_m_tcn_1_residual_block_0_conv1d_1_bias: S
Eread_33_disablecopyonread_adam_v_tcn_1_residual_block_0_conv1d_1_bias: d
Nread_34_disablecopyonread_adam_m_tcn_1_residual_block_0_matching_conv1d_kernel: d
Nread_35_disablecopyonread_adam_v_tcn_1_residual_block_0_matching_conv1d_kernel: Z
Lread_36_disablecopyonread_adam_m_tcn_1_residual_block_0_matching_conv1d_bias: Z
Lread_37_disablecopyonread_adam_v_tcn_1_residual_block_0_matching_conv1d_bias: ]
Gread_38_disablecopyonread_adam_m_tcn_1_residual_block_1_conv1d_0_kernel:  ]
Gread_39_disablecopyonread_adam_v_tcn_1_residual_block_1_conv1d_0_kernel:  S
Eread_40_disablecopyonread_adam_m_tcn_1_residual_block_1_conv1d_0_bias: S
Eread_41_disablecopyonread_adam_v_tcn_1_residual_block_1_conv1d_0_bias: ]
Gread_42_disablecopyonread_adam_m_tcn_1_residual_block_1_conv1d_1_kernel:  ]
Gread_43_disablecopyonread_adam_v_tcn_1_residual_block_1_conv1d_1_kernel:  S
Eread_44_disablecopyonread_adam_m_tcn_1_residual_block_1_conv1d_1_bias: S
Eread_45_disablecopyonread_adam_v_tcn_1_residual_block_1_conv1d_1_bias: ]
Gread_46_disablecopyonread_adam_m_tcn_1_residual_block_2_conv1d_0_kernel:  ]
Gread_47_disablecopyonread_adam_v_tcn_1_residual_block_2_conv1d_0_kernel:  S
Eread_48_disablecopyonread_adam_m_tcn_1_residual_block_2_conv1d_0_bias: S
Eread_49_disablecopyonread_adam_v_tcn_1_residual_block_2_conv1d_0_bias: ]
Gread_50_disablecopyonread_adam_m_tcn_1_residual_block_2_conv1d_1_kernel:  ]
Gread_51_disablecopyonread_adam_v_tcn_1_residual_block_2_conv1d_1_kernel:  S
Eread_52_disablecopyonread_adam_m_tcn_1_residual_block_2_conv1d_1_bias: S
Eread_53_disablecopyonread_adam_v_tcn_1_residual_block_2_conv1d_1_bias: ]
Gread_54_disablecopyonread_adam_m_tcn_1_residual_block_3_conv1d_0_kernel:  ]
Gread_55_disablecopyonread_adam_v_tcn_1_residual_block_3_conv1d_0_kernel:  S
Eread_56_disablecopyonread_adam_m_tcn_1_residual_block_3_conv1d_0_bias: S
Eread_57_disablecopyonread_adam_v_tcn_1_residual_block_3_conv1d_0_bias: ]
Gread_58_disablecopyonread_adam_m_tcn_1_residual_block_3_conv1d_1_kernel:  ]
Gread_59_disablecopyonread_adam_v_tcn_1_residual_block_3_conv1d_1_kernel:  S
Eread_60_disablecopyonread_adam_m_tcn_1_residual_block_3_conv1d_1_bias: S
Eread_61_disablecopyonread_adam_v_tcn_1_residual_block_3_conv1d_1_bias: A
/read_62_disablecopyonread_adam_m_dense_1_kernel:  A
/read_63_disablecopyonread_adam_v_dense_1_kernel:  ;
-read_64_disablecopyonread_adam_m_dense_1_bias: ;
-read_65_disablecopyonread_adam_v_dense_1_bias: A
/read_66_disablecopyonread_adam_m_dense_2_kernel:  A
/read_67_disablecopyonread_adam_v_dense_2_kernel:  ;
-read_68_disablecopyonread_adam_m_dense_2_bias: ;
-read_69_disablecopyonread_adam_v_dense_2_bias: @
.read_70_disablecopyonread_adam_m_output_kernel: @
.read_71_disablecopyonread_adam_v_output_kernel: :
,read_72_disablecopyonread_adam_m_output_bias::
,read_73_disablecopyonread_adam_v_output_bias:+
!read_74_disablecopyonread_total_1: +
!read_75_disablecopyonread_count_1: )
read_76_disablecopyonread_total: )
read_77_disablecopyonread_count: 
savev2_const
identity_157ИҐMergeV2CheckpointsҐRead/DisableCopyOnReadҐRead/ReadVariableOpҐRead_1/DisableCopyOnReadҐRead_1/ReadVariableOpҐRead_10/DisableCopyOnReadҐRead_10/ReadVariableOpҐRead_11/DisableCopyOnReadҐRead_11/ReadVariableOpҐRead_12/DisableCopyOnReadҐRead_12/ReadVariableOpҐRead_13/DisableCopyOnReadҐRead_13/ReadVariableOpҐRead_14/DisableCopyOnReadҐRead_14/ReadVariableOpҐRead_15/DisableCopyOnReadҐRead_15/ReadVariableOpҐRead_16/DisableCopyOnReadҐRead_16/ReadVariableOpҐRead_17/DisableCopyOnReadҐRead_17/ReadVariableOpҐRead_18/DisableCopyOnReadҐRead_18/ReadVariableOpҐRead_19/DisableCopyOnReadҐRead_19/ReadVariableOpҐRead_2/DisableCopyOnReadҐRead_2/ReadVariableOpҐRead_20/DisableCopyOnReadҐRead_20/ReadVariableOpҐRead_21/DisableCopyOnReadҐRead_21/ReadVariableOpҐRead_22/DisableCopyOnReadҐRead_22/ReadVariableOpҐRead_23/DisableCopyOnReadҐRead_23/ReadVariableOpҐRead_24/DisableCopyOnReadҐRead_24/ReadVariableOpҐRead_25/DisableCopyOnReadҐRead_25/ReadVariableOpҐRead_26/DisableCopyOnReadҐRead_26/ReadVariableOpҐRead_27/DisableCopyOnReadҐRead_27/ReadVariableOpҐRead_28/DisableCopyOnReadҐRead_28/ReadVariableOpҐRead_29/DisableCopyOnReadҐRead_29/ReadVariableOpҐRead_3/DisableCopyOnReadҐRead_3/ReadVariableOpҐRead_30/DisableCopyOnReadҐRead_30/ReadVariableOpҐRead_31/DisableCopyOnReadҐRead_31/ReadVariableOpҐRead_32/DisableCopyOnReadҐRead_32/ReadVariableOpҐRead_33/DisableCopyOnReadҐRead_33/ReadVariableOpҐRead_34/DisableCopyOnReadҐRead_34/ReadVariableOpҐRead_35/DisableCopyOnReadҐRead_35/ReadVariableOpҐRead_36/DisableCopyOnReadҐRead_36/ReadVariableOpҐRead_37/DisableCopyOnReadҐRead_37/ReadVariableOpҐRead_38/DisableCopyOnReadҐRead_38/ReadVariableOpҐRead_39/DisableCopyOnReadҐRead_39/ReadVariableOpҐRead_4/DisableCopyOnReadҐRead_4/ReadVariableOpҐRead_40/DisableCopyOnReadҐRead_40/ReadVariableOpҐRead_41/DisableCopyOnReadҐRead_41/ReadVariableOpҐRead_42/DisableCopyOnReadҐRead_42/ReadVariableOpҐRead_43/DisableCopyOnReadҐRead_43/ReadVariableOpҐRead_44/DisableCopyOnReadҐRead_44/ReadVariableOpҐRead_45/DisableCopyOnReadҐRead_45/ReadVariableOpҐRead_46/DisableCopyOnReadҐRead_46/ReadVariableOpҐRead_47/DisableCopyOnReadҐRead_47/ReadVariableOpҐRead_48/DisableCopyOnReadҐRead_48/ReadVariableOpҐRead_49/DisableCopyOnReadҐRead_49/ReadVariableOpҐRead_5/DisableCopyOnReadҐRead_5/ReadVariableOpҐRead_50/DisableCopyOnReadҐRead_50/ReadVariableOpҐRead_51/DisableCopyOnReadҐRead_51/ReadVariableOpҐRead_52/DisableCopyOnReadҐRead_52/ReadVariableOpҐRead_53/DisableCopyOnReadҐRead_53/ReadVariableOpҐRead_54/DisableCopyOnReadҐRead_54/ReadVariableOpҐRead_55/DisableCopyOnReadҐRead_55/ReadVariableOpҐRead_56/DisableCopyOnReadҐRead_56/ReadVariableOpҐRead_57/DisableCopyOnReadҐRead_57/ReadVariableOpҐRead_58/DisableCopyOnReadҐRead_58/ReadVariableOpҐRead_59/DisableCopyOnReadҐRead_59/ReadVariableOpҐRead_6/DisableCopyOnReadҐRead_6/ReadVariableOpҐRead_60/DisableCopyOnReadҐRead_60/ReadVariableOpҐRead_61/DisableCopyOnReadҐRead_61/ReadVariableOpҐRead_62/DisableCopyOnReadҐRead_62/ReadVariableOpҐRead_63/DisableCopyOnReadҐRead_63/ReadVariableOpҐRead_64/DisableCopyOnReadҐRead_64/ReadVariableOpҐRead_65/DisableCopyOnReadҐRead_65/ReadVariableOpҐRead_66/DisableCopyOnReadҐRead_66/ReadVariableOpҐRead_67/DisableCopyOnReadҐRead_67/ReadVariableOpҐRead_68/DisableCopyOnReadҐRead_68/ReadVariableOpҐRead_69/DisableCopyOnReadҐRead_69/ReadVariableOpҐRead_7/DisableCopyOnReadҐRead_7/ReadVariableOpҐRead_70/DisableCopyOnReadҐRead_70/ReadVariableOpҐRead_71/DisableCopyOnReadҐRead_71/ReadVariableOpҐRead_72/DisableCopyOnReadҐRead_72/ReadVariableOpҐRead_73/DisableCopyOnReadҐRead_73/ReadVariableOpҐRead_74/DisableCopyOnReadҐRead_74/ReadVariableOpҐRead_75/DisableCopyOnReadҐRead_75/ReadVariableOpҐRead_76/DisableCopyOnReadҐRead_76/ReadVariableOpҐRead_77/DisableCopyOnReadҐRead_77/ReadVariableOpҐRead_8/DisableCopyOnReadҐRead_8/ReadVariableOpҐRead_9/DisableCopyOnReadҐRead_9/ReadVariableOpw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: w
Read/DisableCopyOnReadDisableCopyOnRead%read_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 °
Read/ReadVariableOpReadVariableOp%read_disablecopyonread_dense_1_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:  y
Read_1/DisableCopyOnReadDisableCopyOnRead%read_1_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 °
Read_1/ReadVariableOpReadVariableOp%read_1_disablecopyonread_dense_1_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: {
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 І
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_2_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:  y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 °
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_2_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: z
Read_4/DisableCopyOnReadDisableCopyOnRead&read_4_disablecopyonread_output_kernel"/device:CPU:0*
_output_shapes
 ¶
Read_4/ReadVariableOpReadVariableOp&read_4_disablecopyonread_output_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

: x
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_output_bias"/device:CPU:0*
_output_shapes
 †
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
:У
Read_6/DisableCopyOnReadDisableCopyOnRead?read_6_disablecopyonread_tcn_1_residual_block_0_conv1d_0_kernel"/device:CPU:0*
_output_shapes
 √
Read_6/ReadVariableOpReadVariableOp?read_6_disablecopyonread_tcn_1_residual_block_0_conv1d_0_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0r
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: i
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*"
_output_shapes
: С
Read_7/DisableCopyOnReadDisableCopyOnRead=read_7_disablecopyonread_tcn_1_residual_block_0_conv1d_0_bias"/device:CPU:0*
_output_shapes
 є
Read_7/ReadVariableOpReadVariableOp=read_7_disablecopyonread_tcn_1_residual_block_0_conv1d_0_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: У
Read_8/DisableCopyOnReadDisableCopyOnRead?read_8_disablecopyonread_tcn_1_residual_block_0_conv1d_1_kernel"/device:CPU:0*
_output_shapes
 √
Read_8/ReadVariableOpReadVariableOp?read_8_disablecopyonread_tcn_1_residual_block_0_conv1d_1_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0r
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  i
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*"
_output_shapes
:  С
Read_9/DisableCopyOnReadDisableCopyOnRead=read_9_disablecopyonread_tcn_1_residual_block_0_conv1d_1_bias"/device:CPU:0*
_output_shapes
 є
Read_9/ReadVariableOpReadVariableOp=read_9_disablecopyonread_tcn_1_residual_block_0_conv1d_1_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: Ь
Read_10/DisableCopyOnReadDisableCopyOnReadGread_10_disablecopyonread_tcn_1_residual_block_0_matching_conv1d_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_10/ReadVariableOpReadVariableOpGread_10_disablecopyonread_tcn_1_residual_block_0_matching_conv1d_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0s
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: i
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*"
_output_shapes
: Ъ
Read_11/DisableCopyOnReadDisableCopyOnReadEread_11_disablecopyonread_tcn_1_residual_block_0_matching_conv1d_bias"/device:CPU:0*
_output_shapes
 √
Read_11/ReadVariableOpReadVariableOpEread_11_disablecopyonread_tcn_1_residual_block_0_matching_conv1d_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: Х
Read_12/DisableCopyOnReadDisableCopyOnRead@read_12_disablecopyonread_tcn_1_residual_block_1_conv1d_0_kernel"/device:CPU:0*
_output_shapes
 ∆
Read_12/ReadVariableOpReadVariableOp@read_12_disablecopyonread_tcn_1_residual_block_1_conv1d_0_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0s
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  i
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*"
_output_shapes
:  У
Read_13/DisableCopyOnReadDisableCopyOnRead>read_13_disablecopyonread_tcn_1_residual_block_1_conv1d_0_bias"/device:CPU:0*
_output_shapes
 Љ
Read_13/ReadVariableOpReadVariableOp>read_13_disablecopyonread_tcn_1_residual_block_1_conv1d_0_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: Х
Read_14/DisableCopyOnReadDisableCopyOnRead@read_14_disablecopyonread_tcn_1_residual_block_1_conv1d_1_kernel"/device:CPU:0*
_output_shapes
 ∆
Read_14/ReadVariableOpReadVariableOp@read_14_disablecopyonread_tcn_1_residual_block_1_conv1d_1_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0s
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  i
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*"
_output_shapes
:  У
Read_15/DisableCopyOnReadDisableCopyOnRead>read_15_disablecopyonread_tcn_1_residual_block_1_conv1d_1_bias"/device:CPU:0*
_output_shapes
 Љ
Read_15/ReadVariableOpReadVariableOp>read_15_disablecopyonread_tcn_1_residual_block_1_conv1d_1_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: Х
Read_16/DisableCopyOnReadDisableCopyOnRead@read_16_disablecopyonread_tcn_1_residual_block_2_conv1d_0_kernel"/device:CPU:0*
_output_shapes
 ∆
Read_16/ReadVariableOpReadVariableOp@read_16_disablecopyonread_tcn_1_residual_block_2_conv1d_0_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0s
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  i
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*"
_output_shapes
:  У
Read_17/DisableCopyOnReadDisableCopyOnRead>read_17_disablecopyonread_tcn_1_residual_block_2_conv1d_0_bias"/device:CPU:0*
_output_shapes
 Љ
Read_17/ReadVariableOpReadVariableOp>read_17_disablecopyonread_tcn_1_residual_block_2_conv1d_0_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: Х
Read_18/DisableCopyOnReadDisableCopyOnRead@read_18_disablecopyonread_tcn_1_residual_block_2_conv1d_1_kernel"/device:CPU:0*
_output_shapes
 ∆
Read_18/ReadVariableOpReadVariableOp@read_18_disablecopyonread_tcn_1_residual_block_2_conv1d_1_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0s
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  i
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*"
_output_shapes
:  У
Read_19/DisableCopyOnReadDisableCopyOnRead>read_19_disablecopyonread_tcn_1_residual_block_2_conv1d_1_bias"/device:CPU:0*
_output_shapes
 Љ
Read_19/ReadVariableOpReadVariableOp>read_19_disablecopyonread_tcn_1_residual_block_2_conv1d_1_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: Х
Read_20/DisableCopyOnReadDisableCopyOnRead@read_20_disablecopyonread_tcn_1_residual_block_3_conv1d_0_kernel"/device:CPU:0*
_output_shapes
 ∆
Read_20/ReadVariableOpReadVariableOp@read_20_disablecopyonread_tcn_1_residual_block_3_conv1d_0_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0s
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  i
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*"
_output_shapes
:  У
Read_21/DisableCopyOnReadDisableCopyOnRead>read_21_disablecopyonread_tcn_1_residual_block_3_conv1d_0_bias"/device:CPU:0*
_output_shapes
 Љ
Read_21/ReadVariableOpReadVariableOp>read_21_disablecopyonread_tcn_1_residual_block_3_conv1d_0_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: Х
Read_22/DisableCopyOnReadDisableCopyOnRead@read_22_disablecopyonread_tcn_1_residual_block_3_conv1d_1_kernel"/device:CPU:0*
_output_shapes
 ∆
Read_22/ReadVariableOpReadVariableOp@read_22_disablecopyonread_tcn_1_residual_block_3_conv1d_1_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0s
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  i
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*"
_output_shapes
:  У
Read_23/DisableCopyOnReadDisableCopyOnRead>read_23_disablecopyonread_tcn_1_residual_block_3_conv1d_1_bias"/device:CPU:0*
_output_shapes
 Љ
Read_23/ReadVariableOpReadVariableOp>read_23_disablecopyonread_tcn_1_residual_block_3_conv1d_1_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_24/DisableCopyOnReadDisableCopyOnRead#read_24_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 Э
Read_24/ReadVariableOpReadVariableOp#read_24_disablecopyonread_iteration^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_25/DisableCopyOnReadDisableCopyOnRead'read_25_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 °
Read_25/ReadVariableOpReadVariableOp'read_25_disablecopyonread_learning_rate^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: Ь
Read_26/DisableCopyOnReadDisableCopyOnReadGread_26_disablecopyonread_adam_m_tcn_1_residual_block_0_conv1d_0_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_26/ReadVariableOpReadVariableOpGread_26_disablecopyonread_adam_m_tcn_1_residual_block_0_conv1d_0_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0s
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: i
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*"
_output_shapes
: Ь
Read_27/DisableCopyOnReadDisableCopyOnReadGread_27_disablecopyonread_adam_v_tcn_1_residual_block_0_conv1d_0_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_27/ReadVariableOpReadVariableOpGread_27_disablecopyonread_adam_v_tcn_1_residual_block_0_conv1d_0_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0s
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: i
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*"
_output_shapes
: Ъ
Read_28/DisableCopyOnReadDisableCopyOnReadEread_28_disablecopyonread_adam_m_tcn_1_residual_block_0_conv1d_0_bias"/device:CPU:0*
_output_shapes
 √
Read_28/ReadVariableOpReadVariableOpEread_28_disablecopyonread_adam_m_tcn_1_residual_block_0_conv1d_0_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: Ъ
Read_29/DisableCopyOnReadDisableCopyOnReadEread_29_disablecopyonread_adam_v_tcn_1_residual_block_0_conv1d_0_bias"/device:CPU:0*
_output_shapes
 √
Read_29/ReadVariableOpReadVariableOpEread_29_disablecopyonread_adam_v_tcn_1_residual_block_0_conv1d_0_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: Ь
Read_30/DisableCopyOnReadDisableCopyOnReadGread_30_disablecopyonread_adam_m_tcn_1_residual_block_0_conv1d_1_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_30/ReadVariableOpReadVariableOpGread_30_disablecopyonread_adam_m_tcn_1_residual_block_0_conv1d_1_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0s
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  i
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*"
_output_shapes
:  Ь
Read_31/DisableCopyOnReadDisableCopyOnReadGread_31_disablecopyonread_adam_v_tcn_1_residual_block_0_conv1d_1_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_31/ReadVariableOpReadVariableOpGread_31_disablecopyonread_adam_v_tcn_1_residual_block_0_conv1d_1_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0s
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  i
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*"
_output_shapes
:  Ъ
Read_32/DisableCopyOnReadDisableCopyOnReadEread_32_disablecopyonread_adam_m_tcn_1_residual_block_0_conv1d_1_bias"/device:CPU:0*
_output_shapes
 √
Read_32/ReadVariableOpReadVariableOpEread_32_disablecopyonread_adam_m_tcn_1_residual_block_0_conv1d_1_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: Ъ
Read_33/DisableCopyOnReadDisableCopyOnReadEread_33_disablecopyonread_adam_v_tcn_1_residual_block_0_conv1d_1_bias"/device:CPU:0*
_output_shapes
 √
Read_33/ReadVariableOpReadVariableOpEread_33_disablecopyonread_adam_v_tcn_1_residual_block_0_conv1d_1_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
: £
Read_34/DisableCopyOnReadDisableCopyOnReadNread_34_disablecopyonread_adam_m_tcn_1_residual_block_0_matching_conv1d_kernel"/device:CPU:0*
_output_shapes
 ‘
Read_34/ReadVariableOpReadVariableOpNread_34_disablecopyonread_adam_m_tcn_1_residual_block_0_matching_conv1d_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0s
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: i
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*"
_output_shapes
: £
Read_35/DisableCopyOnReadDisableCopyOnReadNread_35_disablecopyonread_adam_v_tcn_1_residual_block_0_matching_conv1d_kernel"/device:CPU:0*
_output_shapes
 ‘
Read_35/ReadVariableOpReadVariableOpNread_35_disablecopyonread_adam_v_tcn_1_residual_block_0_matching_conv1d_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
: *
dtype0s
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
: i
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*"
_output_shapes
: °
Read_36/DisableCopyOnReadDisableCopyOnReadLread_36_disablecopyonread_adam_m_tcn_1_residual_block_0_matching_conv1d_bias"/device:CPU:0*
_output_shapes
  
Read_36/ReadVariableOpReadVariableOpLread_36_disablecopyonread_adam_m_tcn_1_residual_block_0_matching_conv1d_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
: °
Read_37/DisableCopyOnReadDisableCopyOnReadLread_37_disablecopyonread_adam_v_tcn_1_residual_block_0_matching_conv1d_bias"/device:CPU:0*
_output_shapes
  
Read_37/ReadVariableOpReadVariableOpLread_37_disablecopyonread_adam_v_tcn_1_residual_block_0_matching_conv1d_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
: Ь
Read_38/DisableCopyOnReadDisableCopyOnReadGread_38_disablecopyonread_adam_m_tcn_1_residual_block_1_conv1d_0_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_38/ReadVariableOpReadVariableOpGread_38_disablecopyonread_adam_m_tcn_1_residual_block_1_conv1d_0_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0s
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  i
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*"
_output_shapes
:  Ь
Read_39/DisableCopyOnReadDisableCopyOnReadGread_39_disablecopyonread_adam_v_tcn_1_residual_block_1_conv1d_0_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_39/ReadVariableOpReadVariableOpGread_39_disablecopyonread_adam_v_tcn_1_residual_block_1_conv1d_0_kernel^Read_39/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0s
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  i
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*"
_output_shapes
:  Ъ
Read_40/DisableCopyOnReadDisableCopyOnReadEread_40_disablecopyonread_adam_m_tcn_1_residual_block_1_conv1d_0_bias"/device:CPU:0*
_output_shapes
 √
Read_40/ReadVariableOpReadVariableOpEread_40_disablecopyonread_adam_m_tcn_1_residual_block_1_conv1d_0_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
: Ъ
Read_41/DisableCopyOnReadDisableCopyOnReadEread_41_disablecopyonread_adam_v_tcn_1_residual_block_1_conv1d_0_bias"/device:CPU:0*
_output_shapes
 √
Read_41/ReadVariableOpReadVariableOpEread_41_disablecopyonread_adam_v_tcn_1_residual_block_1_conv1d_0_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
: Ь
Read_42/DisableCopyOnReadDisableCopyOnReadGread_42_disablecopyonread_adam_m_tcn_1_residual_block_1_conv1d_1_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_42/ReadVariableOpReadVariableOpGread_42_disablecopyonread_adam_m_tcn_1_residual_block_1_conv1d_1_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0s
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  i
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*"
_output_shapes
:  Ь
Read_43/DisableCopyOnReadDisableCopyOnReadGread_43_disablecopyonread_adam_v_tcn_1_residual_block_1_conv1d_1_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_43/ReadVariableOpReadVariableOpGread_43_disablecopyonread_adam_v_tcn_1_residual_block_1_conv1d_1_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0s
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  i
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*"
_output_shapes
:  Ъ
Read_44/DisableCopyOnReadDisableCopyOnReadEread_44_disablecopyonread_adam_m_tcn_1_residual_block_1_conv1d_1_bias"/device:CPU:0*
_output_shapes
 √
Read_44/ReadVariableOpReadVariableOpEread_44_disablecopyonread_adam_m_tcn_1_residual_block_1_conv1d_1_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
: Ъ
Read_45/DisableCopyOnReadDisableCopyOnReadEread_45_disablecopyonread_adam_v_tcn_1_residual_block_1_conv1d_1_bias"/device:CPU:0*
_output_shapes
 √
Read_45/ReadVariableOpReadVariableOpEread_45_disablecopyonread_adam_v_tcn_1_residual_block_1_conv1d_1_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
: Ь
Read_46/DisableCopyOnReadDisableCopyOnReadGread_46_disablecopyonread_adam_m_tcn_1_residual_block_2_conv1d_0_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_46/ReadVariableOpReadVariableOpGread_46_disablecopyonread_adam_m_tcn_1_residual_block_2_conv1d_0_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0s
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  i
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*"
_output_shapes
:  Ь
Read_47/DisableCopyOnReadDisableCopyOnReadGread_47_disablecopyonread_adam_v_tcn_1_residual_block_2_conv1d_0_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_47/ReadVariableOpReadVariableOpGread_47_disablecopyonread_adam_v_tcn_1_residual_block_2_conv1d_0_kernel^Read_47/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0s
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  i
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*"
_output_shapes
:  Ъ
Read_48/DisableCopyOnReadDisableCopyOnReadEread_48_disablecopyonread_adam_m_tcn_1_residual_block_2_conv1d_0_bias"/device:CPU:0*
_output_shapes
 √
Read_48/ReadVariableOpReadVariableOpEread_48_disablecopyonread_adam_m_tcn_1_residual_block_2_conv1d_0_bias^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
: Ъ
Read_49/DisableCopyOnReadDisableCopyOnReadEread_49_disablecopyonread_adam_v_tcn_1_residual_block_2_conv1d_0_bias"/device:CPU:0*
_output_shapes
 √
Read_49/ReadVariableOpReadVariableOpEread_49_disablecopyonread_adam_v_tcn_1_residual_block_2_conv1d_0_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
: Ь
Read_50/DisableCopyOnReadDisableCopyOnReadGread_50_disablecopyonread_adam_m_tcn_1_residual_block_2_conv1d_1_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_50/ReadVariableOpReadVariableOpGread_50_disablecopyonread_adam_m_tcn_1_residual_block_2_conv1d_1_kernel^Read_50/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0t
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  k
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*"
_output_shapes
:  Ь
Read_51/DisableCopyOnReadDisableCopyOnReadGread_51_disablecopyonread_adam_v_tcn_1_residual_block_2_conv1d_1_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_51/ReadVariableOpReadVariableOpGread_51_disablecopyonread_adam_v_tcn_1_residual_block_2_conv1d_1_kernel^Read_51/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0t
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  k
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*"
_output_shapes
:  Ъ
Read_52/DisableCopyOnReadDisableCopyOnReadEread_52_disablecopyonread_adam_m_tcn_1_residual_block_2_conv1d_1_bias"/device:CPU:0*
_output_shapes
 √
Read_52/ReadVariableOpReadVariableOpEread_52_disablecopyonread_adam_m_tcn_1_residual_block_2_conv1d_1_bias^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
: Ъ
Read_53/DisableCopyOnReadDisableCopyOnReadEread_53_disablecopyonread_adam_v_tcn_1_residual_block_2_conv1d_1_bias"/device:CPU:0*
_output_shapes
 √
Read_53/ReadVariableOpReadVariableOpEread_53_disablecopyonread_adam_v_tcn_1_residual_block_2_conv1d_1_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
: Ь
Read_54/DisableCopyOnReadDisableCopyOnReadGread_54_disablecopyonread_adam_m_tcn_1_residual_block_3_conv1d_0_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_54/ReadVariableOpReadVariableOpGread_54_disablecopyonread_adam_m_tcn_1_residual_block_3_conv1d_0_kernel^Read_54/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0t
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  k
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*"
_output_shapes
:  Ь
Read_55/DisableCopyOnReadDisableCopyOnReadGread_55_disablecopyonread_adam_v_tcn_1_residual_block_3_conv1d_0_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_55/ReadVariableOpReadVariableOpGread_55_disablecopyonread_adam_v_tcn_1_residual_block_3_conv1d_0_kernel^Read_55/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0t
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  k
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*"
_output_shapes
:  Ъ
Read_56/DisableCopyOnReadDisableCopyOnReadEread_56_disablecopyonread_adam_m_tcn_1_residual_block_3_conv1d_0_bias"/device:CPU:0*
_output_shapes
 √
Read_56/ReadVariableOpReadVariableOpEread_56_disablecopyonread_adam_m_tcn_1_residual_block_3_conv1d_0_bias^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
: Ъ
Read_57/DisableCopyOnReadDisableCopyOnReadEread_57_disablecopyonread_adam_v_tcn_1_residual_block_3_conv1d_0_bias"/device:CPU:0*
_output_shapes
 √
Read_57/ReadVariableOpReadVariableOpEread_57_disablecopyonread_adam_v_tcn_1_residual_block_3_conv1d_0_bias^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
: Ь
Read_58/DisableCopyOnReadDisableCopyOnReadGread_58_disablecopyonread_adam_m_tcn_1_residual_block_3_conv1d_1_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_58/ReadVariableOpReadVariableOpGread_58_disablecopyonread_adam_m_tcn_1_residual_block_3_conv1d_1_kernel^Read_58/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0t
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  k
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*"
_output_shapes
:  Ь
Read_59/DisableCopyOnReadDisableCopyOnReadGread_59_disablecopyonread_adam_v_tcn_1_residual_block_3_conv1d_1_kernel"/device:CPU:0*
_output_shapes
 Ќ
Read_59/ReadVariableOpReadVariableOpGread_59_disablecopyonread_adam_v_tcn_1_residual_block_3_conv1d_1_kernel^Read_59/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:  *
dtype0t
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:  k
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*"
_output_shapes
:  Ъ
Read_60/DisableCopyOnReadDisableCopyOnReadEread_60_disablecopyonread_adam_m_tcn_1_residual_block_3_conv1d_1_bias"/device:CPU:0*
_output_shapes
 √
Read_60/ReadVariableOpReadVariableOpEread_60_disablecopyonread_adam_m_tcn_1_residual_block_3_conv1d_1_bias^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
: Ъ
Read_61/DisableCopyOnReadDisableCopyOnReadEread_61_disablecopyonread_adam_v_tcn_1_residual_block_3_conv1d_1_bias"/device:CPU:0*
_output_shapes
 √
Read_61/ReadVariableOpReadVariableOpEread_61_disablecopyonread_adam_v_tcn_1_residual_block_3_conv1d_1_bias^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
: Д
Read_62/DisableCopyOnReadDisableCopyOnRead/read_62_disablecopyonread_adam_m_dense_1_kernel"/device:CPU:0*
_output_shapes
 ±
Read_62/ReadVariableOpReadVariableOp/read_62_disablecopyonread_adam_m_dense_1_kernel^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0p
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  g
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes

:  Д
Read_63/DisableCopyOnReadDisableCopyOnRead/read_63_disablecopyonread_adam_v_dense_1_kernel"/device:CPU:0*
_output_shapes
 ±
Read_63/ReadVariableOpReadVariableOp/read_63_disablecopyonread_adam_v_dense_1_kernel^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0p
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  g
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes

:  В
Read_64/DisableCopyOnReadDisableCopyOnRead-read_64_disablecopyonread_adam_m_dense_1_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_64/ReadVariableOpReadVariableOp-read_64_disablecopyonread_adam_m_dense_1_bias^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes
: В
Read_65/DisableCopyOnReadDisableCopyOnRead-read_65_disablecopyonread_adam_v_dense_1_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_65/ReadVariableOpReadVariableOp-read_65_disablecopyonread_adam_v_dense_1_bias^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes
: Д
Read_66/DisableCopyOnReadDisableCopyOnRead/read_66_disablecopyonread_adam_m_dense_2_kernel"/device:CPU:0*
_output_shapes
 ±
Read_66/ReadVariableOpReadVariableOp/read_66_disablecopyonread_adam_m_dense_2_kernel^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0p
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  g
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes

:  Д
Read_67/DisableCopyOnReadDisableCopyOnRead/read_67_disablecopyonread_adam_v_dense_2_kernel"/device:CPU:0*
_output_shapes
 ±
Read_67/ReadVariableOpReadVariableOp/read_67_disablecopyonread_adam_v_dense_2_kernel^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0p
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  g
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes

:  В
Read_68/DisableCopyOnReadDisableCopyOnRead-read_68_disablecopyonread_adam_m_dense_2_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_68/ReadVariableOpReadVariableOp-read_68_disablecopyonread_adam_m_dense_2_bias^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes
: В
Read_69/DisableCopyOnReadDisableCopyOnRead-read_69_disablecopyonread_adam_v_dense_2_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_69/ReadVariableOpReadVariableOp-read_69_disablecopyonread_adam_v_dense_2_bias^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
: Г
Read_70/DisableCopyOnReadDisableCopyOnRead.read_70_disablecopyonread_adam_m_output_kernel"/device:CPU:0*
_output_shapes
 ∞
Read_70/ReadVariableOpReadVariableOp.read_70_disablecopyonread_adam_m_output_kernel^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0p
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: g
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes

: Г
Read_71/DisableCopyOnReadDisableCopyOnRead.read_71_disablecopyonread_adam_v_output_kernel"/device:CPU:0*
_output_shapes
 ∞
Read_71/ReadVariableOpReadVariableOp.read_71_disablecopyonread_adam_v_output_kernel^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0p
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: g
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes

: Б
Read_72/DisableCopyOnReadDisableCopyOnRead,read_72_disablecopyonread_adam_m_output_bias"/device:CPU:0*
_output_shapes
 ™
Read_72/ReadVariableOpReadVariableOp,read_72_disablecopyonread_adam_m_output_bias^Read_72/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_73/DisableCopyOnReadDisableCopyOnRead,read_73_disablecopyonread_adam_v_output_bias"/device:CPU:0*
_output_shapes
 ™
Read_73/ReadVariableOpReadVariableOp,read_73_disablecopyonread_adam_v_output_bias^Read_73/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_74/DisableCopyOnReadDisableCopyOnRead!read_74_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 Ы
Read_74/ReadVariableOpReadVariableOp!read_74_disablecopyonread_total_1^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_75/DisableCopyOnReadDisableCopyOnRead!read_75_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 Ы
Read_75/ReadVariableOpReadVariableOp!read_75_disablecopyonread_count_1^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_76/DisableCopyOnReadDisableCopyOnReadread_76_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Щ
Read_76/ReadVariableOpReadVariableOpread_76_disablecopyonread_total^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_77/DisableCopyOnReadDisableCopyOnReadread_77_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Щ
Read_77/ReadVariableOpReadVariableOpread_77_disablecopyonread_count^Read_77/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes
: Ґ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*Ћ
valueЅBЊOB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHО
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*≥
value©B¶OB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B с
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *]
dtypesS
Q2O	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:≥
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_156Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_157IdentityIdentity_156:output:0^NoOp*
T0*
_output_shapes
: й 
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "%
identity_157Identity_157:output:0*µ
_input_shapes£
†: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:O

_output_shapes
: 
н
¶
__inference_loss_fn_3_15567E
7dense_2_bias_regularizer_l2loss_readvariableop_resource: 
identityИҐ.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpҐ
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp7dense_2_bias_regularizer_l2loss_readvariableop_resource*
_output_shapes
: *
dtype0В
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
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
Ц
н
%__inference_TCN_1_layer_call_fn_14966

inputs
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9:  

unknown_10:  

unknown_11:  

unknown_12:  

unknown_13:  

unknown_14:  

unknown_15:  

unknown_16: 
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_TCN_1_layer_call_and_return_conditional_losses_13315o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€В: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€В
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15600

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_12926

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ћ
‘
A__inference_Output_layer_call_and_return_conditional_losses_15531

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ-Output/bias/Regularizer/L2Loss/ReadVariableOpҐ/Output/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€О
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ъ
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Й
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0А
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ф
Output/bias/Regularizer/mulMul&Output/bias/Regularizer/mul/x:output:0'Output/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ў
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^Output/bias/Regularizer/L2Loss/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-Output/bias/Regularizer/L2Loss/ReadVariableOp-Output/bias/Regularizer/L2Loss/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
к∆
”
@__inference_TCN_1_layer_call_and_return_conditional_losses_13675

inputs[
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource: G
9residual_block_0_conv1d_0_biasadd_readvariableop_resource: [
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_0_conv1d_1_biasadd_readvariableop_resource: b
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource: N
@residual_block_0_matching_conv1d_biasadd_readvariableop_resource: [
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_1_conv1d_0_biasadd_readvariableop_resource: [
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_1_conv1d_1_biasadd_readvariableop_resource: [
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_2_conv1d_0_biasadd_readvariableop_resource: [
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_2_conv1d_1_biasadd_readvariableop_resource: [
Eresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_3_conv1d_0_biasadd_readvariableop_resource: [
Eresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  G
9residual_block_3_conv1d_1_biasadd_readvariableop_resource: 
identityИҐ0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpҐ<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpҐ<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpҐCresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpҐ<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpҐ<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpҐ<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpҐ<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpҐ<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpҐ0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpҐ<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpП
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Ф
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Дz
/residual_block_0/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€÷
+residual_block_0/conv1D_0/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_0/Pad:output:08residual_block_0/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Д∆
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0s
1residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_0/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ь
 residual_block_0/conv1D_0/Conv1DConv2D4residual_block_0/conv1D_0/Conv1D/ExpandDims:output:06residual_block_0/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingVALID*
strides
µ
(residual_block_0/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_0/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€¶
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0–
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/Conv1D/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_0/Act_Conv1D_0/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_0/SDropout_0/IdentityIdentity0residual_block_0/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ї
residual_block_0/conv1D_1/PadPad-residual_block_0/SDropout_0/Identity:output:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Д z
/residual_block_0/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€÷
+residual_block_0/conv1D_1/Conv1D/ExpandDims
ExpandDims&residual_block_0/conv1D_1/Pad:output:08residual_block_0/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Д ∆
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_0/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ь
 residual_block_0/conv1D_1/Conv1DConv2D4residual_block_0/conv1D_1/Conv1D/ExpandDims:output:06residual_block_0/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingVALID*
strides
µ
(residual_block_0/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_0/conv1D_1/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€¶
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0–
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/Conv1D/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_0/Act_Conv1D_1/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_0/SDropout_1/IdentityIdentity0residual_block_0/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В У
%residual_block_0/Act_Conv_Blocks/ReluRelu-residual_block_0/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Б
6residual_block_0/matching_conv1D/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€ƒ
2residual_block_0/matching_conv1D/Conv1D/ExpandDims
ExpandDimsinputs?residual_block_0/matching_conv1D/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€В‘
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpLresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0z
8residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Г
4residual_block_0/matching_conv1D/Conv1D/ExpandDims_1
ExpandDimsKresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp:value:0Aresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Р
'residual_block_0/matching_conv1D/Conv1DConv2D;residual_block_0/matching_conv1D/Conv1D/ExpandDims:output:0=residual_block_0/matching_conv1D/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€В *
paddingSAME*
strides
√
/residual_block_0/matching_conv1D/Conv1D/SqueezeSqueeze0residual_block_0/matching_conv1D/Conv1D:output:0*
T0*,
_output_shapes
:€€€€€€€€€В *
squeeze_dims

э€€€€€€€€і
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0е
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/Conv1D/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В ƒ
residual_block_0/Add_Res/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:03residual_block_0/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Д
#residual_block_0/Act_Res_Block/ReluRelu residual_block_0/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       њ
residual_block_1/conv1D_0/PadPad1residual_block_0/Act_Res_Block/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ж x
.residual_block_1/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ж®
Oresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        †
Gresidual_block_1/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        Е
;residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ®
/residual_block_1/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_0/Pad:output:0Dresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€C z
/residual_block_1/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_1/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€C ∆
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_1/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_1/conv1D_0/Conv1DConv2D4residual_block_1/conv1D_0/Conv1D/ExpandDims:output:06residual_block_1/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€A *
paddingVALID*
strides
і
(residual_block_1/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€A *
squeeze_dims

э€€€€€€€€Е
;residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ±
/residual_block_1/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_1/Act_Conv1D_0/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_1/SDropout_0/IdentityIdentity0residual_block_1/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ї
residual_block_1/conv1D_1/PadPad-residual_block_1/SDropout_0/Identity:output:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ж x
.residual_block_1/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ж®
Oresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        †
Gresidual_block_1/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        Е
;residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        ®
/residual_block_1/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_1/Pad:output:0Dresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€C z
/residual_block_1/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_1/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_1/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_1/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€C ∆
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_1/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_1/conv1D_1/Conv1DConv2D4residual_block_1/conv1D_1/Conv1D/ExpandDims:output:06residual_block_1/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€A *
paddingVALID*
strides
і
(residual_block_1/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_1/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€A *
squeeze_dims

э€€€€€€€€Е
;residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        ±
/residual_block_1/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_1/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_1/Act_Conv1D_1/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_1/SDropout_1/IdentityIdentity0residual_block_1/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В У
%residual_block_1/Act_Conv_Blocks/ReluRelu-residual_block_1/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ƒ
residual_block_1/Add_Res/addAddV21residual_block_0/Act_Res_Block/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Д
#residual_block_1/Act_Res_Block/ReluRelu residual_block_1/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       њ
residual_block_2/conv1D_0/PadPad1residual_block_1/Act_Res_Block/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€К x
.residual_block_2/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:К®
Oresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       †
Gresidual_block_2/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Е
;residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ®
/residual_block_2/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_0/Pad:output:0Dresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€# z
/residual_block_2/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_2/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€# ∆
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_2/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_2/conv1D_0/Conv1DConv2D4residual_block_2/conv1D_0/Conv1D/ExpandDims:output:06residual_block_2/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€! *
paddingVALID*
strides
і
(residual_block_2/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€! *
squeeze_dims

э€€€€€€€€Е
;residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ±
/residual_block_2/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_2/Act_Conv1D_0/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_2/SDropout_0/IdentityIdentity0residual_block_2/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ї
residual_block_2/conv1D_1/PadPad-residual_block_2/SDropout_0/Identity:output:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€К x
.residual_block_2/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:К®
Oresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       †
Gresidual_block_2/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Е
;residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ®
/residual_block_2/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_1/Pad:output:0Dresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€# z
/residual_block_2/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_2/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_2/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_2/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€# ∆
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_2/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_2/conv1D_1/Conv1DConv2D4residual_block_2/conv1D_1/Conv1D/ExpandDims:output:06residual_block_2/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€! *
paddingVALID*
strides
і
(residual_block_2/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_2/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€! *
squeeze_dims

э€€€€€€€€Е
;residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ±
/residual_block_2/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_2/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_2/Act_Conv1D_1/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_2/SDropout_1/IdentityIdentity0residual_block_2/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В У
%residual_block_2/Act_Conv_Blocks/ReluRelu-residual_block_2/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ƒ
residual_block_2/Add_Res/addAddV21residual_block_1/Act_Res_Block/Relu:activations:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Д
#residual_block_2/Act_Res_Block/ReluRelu residual_block_2/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_3/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       њ
residual_block_3/conv1D_0/PadPad1residual_block_2/Act_Res_Block/Relu:activations:0/residual_block_3/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Т x
.residual_block_3/conv1D_0/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Т®
Oresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       †
Gresidual_block_3/conv1D_0/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Е
;residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ®
/residual_block_3/conv1D_0/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_0/Pad:output:0Dresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_0/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€ z
/residual_block_3/conv1D_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_3/conv1D_0/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_0/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ∆
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_0_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_3/conv1D_0/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_0/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_3/conv1D_0/Conv1DConv2D4residual_block_3/conv1D_0/Conv1D/ExpandDims:output:06residual_block_3/conv1D_0/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
і
(residual_block_3/conv1D_0/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_0/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€Е
;residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ±
/residual_block_3/conv1D_0/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_0/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_0/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_0/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_3/conv1D_0/BiasAddBiasAdd8residual_block_3/conv1D_0/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_3/Act_Conv1D_0/ReluRelu*residual_block_3/conv1D_0/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_3/SDropout_0/IdentityIdentity0residual_block_3/Act_Conv1D_0/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В П
&residual_block_3/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       ї
residual_block_3/conv1D_1/PadPad-residual_block_3/SDropout_0/Identity:output:0/residual_block_3/conv1D_1/Pad/paddings:output:0*
T0*,
_output_shapes
:€€€€€€€€€Т x
.residual_block_3/conv1D_1/Conv1D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:Ш
Mresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:Т®
Oresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        £
Jresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       †
Gresidual_block_3/conv1D_1/Conv1D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       Е
;residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:С
8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"       ®
/residual_block_3/conv1D_1/Conv1D/SpaceToBatchNDSpaceToBatchND&residual_block_3/conv1D_1/Pad:output:0Dresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/block_shape:output:0Aresidual_block_3/conv1D_1/Conv1D/SpaceToBatchND/paddings:output:0*
T0*+
_output_shapes
:€€€€€€€€€ z
/residual_block_3/conv1D_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€з
+residual_block_3/conv1D_1/Conv1D/ExpandDims
ExpandDims8residual_block_3/conv1D_1/Conv1D/SpaceToBatchND:output:08residual_block_3/conv1D_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ∆
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_3_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : о
-residual_block_3/conv1D_1/Conv1D/ExpandDims_1
ExpandDimsDresidual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0:residual_block_3/conv1D_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ы
 residual_block_3/conv1D_1/Conv1DConv2D4residual_block_3/conv1D_1/Conv1D/ExpandDims:output:06residual_block_3/conv1D_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
і
(residual_block_3/conv1D_1/Conv1D/SqueezeSqueeze)residual_block_3/conv1D_1/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€Е
;residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:О
5residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"       ±
/residual_block_3/conv1D_1/Conv1D/BatchToSpaceNDBatchToSpaceND1residual_block_3/conv1D_1/Conv1D/Squeeze:output:0Dresidual_block_3/conv1D_1/Conv1D/BatchToSpaceND/block_shape:output:0>residual_block_3/conv1D_1/Conv1D/BatchToSpaceND/crops:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ¶
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_3_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0„
!residual_block_3/conv1D_1/BiasAddBiasAdd8residual_block_3/conv1D_1/Conv1D/BatchToSpaceND:output:08residual_block_3/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€В Н
"residual_block_3/Act_Conv1D_1/ReluRelu*residual_block_3/conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€В Щ
$residual_block_3/SDropout_1/IdentityIdentity0residual_block_3/Act_Conv1D_1/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В У
%residual_block_3/Act_Conv_Blocks/ReluRelu-residual_block_3/SDropout_1/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€В ƒ
residual_block_3/Add_Res/addAddV21residual_block_2/Act_Res_Block/Relu:activations:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В Д
#residual_block_3/Act_Res_Block/ReluRelu residual_block_3/Add_Res/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€В ¬
Add_Skip_Connections/addAddV23residual_block_0/Act_Conv_Blocks/Relu:activations:03residual_block_1/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В ≠
Add_Skip_Connections/add_1AddV2Add_Skip_Connections/add:z:03residual_block_2/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В ѓ
Add_Skip_Connections/add_2AddV2Add_Skip_Connections/add_1:z:03residual_block_3/Act_Conv_Blocks/Relu:activations:0*
T0*,
_output_shapes
:€€€€€€€€€В u
 Slice_Output/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    €€€€    w
"Slice_Output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            w
"Slice_Output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         »
Slice_Output/strided_sliceStridedSliceAdd_Skip_Connections/add_2:z:0)Slice_Output/strided_slice/stack:output:0+Slice_Output/strided_slice/stack_1:output:0+Slice_Output/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *

begin_mask*
end_mask*
shrink_axis_maskr
IdentityIdentity#Slice_Output/strided_slice:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ ÷
NoOpNoOp1^residual_block_0/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_0/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp8^residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpD^residual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_1/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_2/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_0/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp1^residual_block_3/conv1D_1/BiasAdd/ReadVariableOp=^residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:€€€€€€€€€В: : : : : : : : : : : : : : : : : : 2d
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp0residual_block_0/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp0residual_block_0/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_0/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2r
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp2К
Cresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOpCresidual_block_0/matching_conv1D/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp0residual_block_1/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp0residual_block_1/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_1/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp0residual_block_2/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp0residual_block_2/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_2/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp0residual_block_3/conv1D_0/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_0/Conv1D/ExpandDims_1/ReadVariableOp2d
0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp0residual_block_3/conv1D_1/BiasAdd/ReadVariableOp2|
<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp<residual_block_3/conv1D_1/Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€В
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15665

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
F
*__inference_SDropout_1_layer_call_fn_15695

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_13036v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ї
°
,__inference_sequential_1_layer_call_fn_13887
tcn_1_input
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9:  

unknown_10:  

unknown_11:  

unknown_12:  

unknown_13:  

unknown_14:  

unknown_15:  

unknown_16: 

unknown_17:  

unknown_18: 

unknown_19:  

unknown_20: 

unknown_21: 

unknown_22:
identityИҐStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCalltcn_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_13836o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€В: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:€€€€€€€€€В
%
_user_specified_nameTCN_1_input
∞
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15685

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Њ
Ф
'__inference_Dense_1_layer_call_fn_15456

inputs
unknown:  
	unknown_0: 
identityИҐStatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_13372o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_13036

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_12921

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Џ=
ы	
G__inference_sequential_1_layer_call_and_return_conditional_losses_13836

inputs!
tcn_1_13759: 
tcn_1_13761: !
tcn_1_13763:  
tcn_1_13765: !
tcn_1_13767: 
tcn_1_13769: !
tcn_1_13771:  
tcn_1_13773: !
tcn_1_13775:  
tcn_1_13777: !
tcn_1_13779:  
tcn_1_13781: !
tcn_1_13783:  
tcn_1_13785: !
tcn_1_13787:  
tcn_1_13789: !
tcn_1_13791:  
tcn_1_13793: 
dense_1_13796:  
dense_1_13798: 
dense_2_13801:  
dense_2_13803: 
output_13806: 
output_13808:
identityИҐDense_1/StatefulPartitionedCallҐ.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpҐ0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpҐDense_2/StatefulPartitionedCallҐ.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpҐ0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpҐOutput/StatefulPartitionedCallҐ-Output/bias/Regularizer/L2Loss/ReadVariableOpҐ/Output/kernel/Regularizer/L2Loss/ReadVariableOpҐTCN_1/StatefulPartitionedCall—
TCN_1/StatefulPartitionedCallStatefulPartitionedCallinputstcn_1_13759tcn_1_13761tcn_1_13763tcn_1_13765tcn_1_13767tcn_1_13769tcn_1_13771tcn_1_13773tcn_1_13775tcn_1_13777tcn_1_13779tcn_1_13781tcn_1_13783tcn_1_13785tcn_1_13787tcn_1_13789tcn_1_13791tcn_1_13793*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_TCN_1_layer_call_and_return_conditional_losses_13315Й
Dense_1/StatefulPartitionedCallStatefulPartitionedCall&TCN_1/StatefulPartitionedCall:output:0dense_1_13796dense_1_13798*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_13372Л
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_13801dense_2_13803*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_13397З
Output/StatefulPartitionedCallStatefulPartitionedCall(Dense_2/StatefulPartitionedCall:output:0output_13806output_13808*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_Output_layer_call_and_return_conditional_losses_13422~
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_13796*
_output_shapes

:  *
dtype0Ж
!Dense_1/kernel/Regularizer/L2LossL2Loss8Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
Dense_1/kernel/Regularizer/mulMul)Dense_1/kernel/Regularizer/mul/x:output:0*Dense_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1_13798*
_output_shapes
: *
dtype0В
Dense_1/bias/Regularizer/L2LossL2Loss6Dense_1/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
Dense_1/bias/Regularizer/mulMul'Dense_1/bias/Regularizer/mul/x:output:0(Dense_1/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ~
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_13801*
_output_shapes

:  *
dtype0Ж
!Dense_2/kernel/Regularizer/L2LossL2Loss8Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 Dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Э
Dense_2/kernel/Regularizer/mulMul)Dense_2/kernel/Regularizer/mul/x:output:0*Dense_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_2_13803*
_output_shapes
: *
dtype0В
Dense_2/bias/Regularizer/L2LossL2Loss6Dense_2/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Dense_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ч
Dense_2/bias/Regularizer/mulMul'Dense_2/bias/Regularizer/mul/x:output:0(Dense_2/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_13806*
_output_shapes

: *
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ъ
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
-Output/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_13808*
_output_shapes
:*
dtype0А
Output/bias/Regularizer/L2LossL2Loss5Output/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: b
Output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ76Ф
Output/bias/Regularizer/mulMul&Output/bias/Regularizer/mul/x:output:0'Output/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€х
NoOpNoOp ^Dense_1/StatefulPartitionedCall/^Dense_1/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp ^Dense_2/StatefulPartitionedCall/^Dense_2/bias/Regularizer/L2Loss/ReadVariableOp1^Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp^Output/StatefulPartitionedCall.^Output/bias/Regularizer/L2Loss/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp^TCN_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€В: : : : : : : : : : : : : : : : : : : : : : : : 2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2`
.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp.Dense_1/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_1/kernel/Regularizer/L2Loss/ReadVariableOp2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2`
.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp.Dense_2/bias/Regularizer/L2Loss/ReadVariableOp2d
0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp0Dense_2/kernel/Regularizer/L2Loss/ReadVariableOp2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2^
-Output/bias/Regularizer/L2Loss/ReadVariableOp-Output/bias/Regularizer/L2Loss/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp2>
TCN_1/StatefulPartitionedCallTCN_1/StatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€В
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15620

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
F
*__inference_SDropout_1_layer_call_fn_15650

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_12987v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞
c
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15640

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
F
*__inference_SDropout_1_layer_call_fn_15730

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_SDropout_1_layer_call_and_return_conditional_losses_13075v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
F
*__inference_SDropout_0_layer_call_fn_15595

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_SDropout_0_layer_call_and_return_conditional_losses_12926v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs"у
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*С
serving_default~
+
x&
serving_default_x:0В3
output_0'
StatefulPartitionedCall:0tensorflow/serving/predict:«ѕ
В
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
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
Ё
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	dilations
skip_connections
residual_blocks
layers_outputs
residual_block_0
residual_block_1
residual_block_2
residual_block_3
slicer_layer"
_tf_keras_layer
ї
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
ї
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias"
_tf_keras_layer
ї
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias"
_tf_keras_layer
÷
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17
#18
$19
+20
,21
322
423"
trackable_list_wrapper
÷
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17
#18
$19
+20
,21
322
423"
trackable_list_wrapper
J
G0
H1
I2
J3
K4
L5"
trackable_list_wrapper
 
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
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
џ
Rtrace_0
Strace_1
Ttrace_2
Utrace_32р
,__inference_sequential_1_layer_call_fn_13887
,__inference_sequential_1_layer_call_fn_14020
,__inference_sequential_1_layer_call_fn_14342
,__inference_sequential_1_layer_call_fn_14395µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zRtrace_0zStrace_1zTtrace_2zUtrace_3
«
Vtrace_0
Wtrace_1
Xtrace_2
Ytrace_32№
G__inference_sequential_1_layer_call_and_return_conditional_losses_13453
G__inference_sequential_1_layer_call_and_return_conditional_losses_13753
G__inference_sequential_1_layer_call_and_return_conditional_losses_14660
G__inference_sequential_1_layer_call_and_return_conditional_losses_14925µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zVtrace_0zWtrace_1zXtrace_2zYtrace_3
ѕBћ
 __inference__wrapped_model_12915TCN_1_input"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ь
Z
_variables
[_iterations
\_learning_rate
]_index_dict
^
_momentums
__velocities
`_update_step_xla"
experimentalOptimizer
,
aserving_default"
signature_map
¶
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17"
trackable_list_wrapper
¶
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
D15
E16
F17"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ј
gtrace_0
htrace_12А
%__inference_TCN_1_layer_call_fn_14966
%__inference_TCN_1_layer_call_fn_15007ѓ
®≤§
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkwjkwargs
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zgtrace_0zhtrace_1
н
itrace_0
jtrace_12ґ
@__inference_TCN_1_layer_call_and_return_conditional_losses_15227
@__inference_TCN_1_layer_call_and_return_conditional_losses_15447ѓ
®≤§
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkwjkwargs
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zitrace_0zjtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
ъ
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

qlayers
rshape_match_conv
sfinal_activation
tconv1D_0
uAct_Conv1D_0
v
SDropout_0
wconv1D_1
xAct_Conv1D_1
y
SDropout_1
zAct_Conv_Blocks
rmatching_conv1D
sAct_Res_Block"
_tf_keras_layer
Й
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+А&call_and_return_all_conditional_losses
Бlayers
Вshape_match_conv
Гfinal_activation
Дconv1D_0
ЕAct_Conv1D_0
Ж
SDropout_0
Зconv1D_1
ИAct_Conv1D_1
Й
SDropout_1
КAct_Conv_Blocks
Вmatching_identity
ГAct_Res_Block"
_tf_keras_layer
О
Л	variables
Мtrainable_variables
Нregularization_losses
О	keras_api
П__call__
+Р&call_and_return_all_conditional_losses
Сlayers
Тshape_match_conv
Уfinal_activation
Фconv1D_0
ХAct_Conv1D_0
Ц
SDropout_0
Чconv1D_1
ШAct_Conv1D_1
Щ
SDropout_1
ЪAct_Conv_Blocks
Тmatching_identity
УAct_Res_Block"
_tf_keras_layer
О
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+†&call_and_return_all_conditional_losses
°layers
Ґshape_match_conv
£final_activation
§conv1D_0
•Act_Conv1D_0
¶
SDropout_0
Іconv1D_1
®Act_Conv1D_1
©
SDropout_1
™Act_Conv_Blocks
Ґmatching_identity
£Act_Res_Block"
_tf_keras_layer
Ђ
Ђ	variables
ђtrainable_variables
≠regularization_losses
Ѓ	keras_api
ѓ__call__
+∞&call_and_return_all_conditional_losses"
_tf_keras_layer
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
≤
±non_trainable_variables
≤layers
≥metrics
 іlayer_regularization_losses
µlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
г
ґtrace_02ƒ
'__inference_Dense_1_layer_call_fn_15456Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zґtrace_0
ю
Јtrace_02я
B__inference_Dense_1_layer_call_and_return_conditional_losses_15475Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЈtrace_0
 :  2Dense_1/kernel
: 2Dense_1/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
≤
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Љlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
г
љtrace_02ƒ
'__inference_Dense_2_layer_call_fn_15484Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zљtrace_0
ю
Њtrace_02я
B__inference_Dense_2_layer_call_and_return_conditional_losses_15503Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЊtrace_0
 :  2Dense_2/kernel
: 2Dense_2/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
≤
њnon_trainable_variables
јlayers
Ѕmetrics
 ¬layer_regularization_losses
√layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
в
ƒtrace_02√
&__inference_Output_layer_call_fn_15512Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zƒtrace_0
э
≈trace_02ё
A__inference_Output_layer_call_and_return_conditional_losses_15531Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z≈trace_0
: 2Output/kernel
:2Output/bias
<:: 2&TCN_1/residual_block_0/conv1D_0/kernel
2:0 2$TCN_1/residual_block_0/conv1D_0/bias
<::  2&TCN_1/residual_block_0/conv1D_1/kernel
2:0 2$TCN_1/residual_block_0/conv1D_1/bias
C:A 2-TCN_1/residual_block_0/matching_conv1D/kernel
9:7 2+TCN_1/residual_block_0/matching_conv1D/bias
<::  2&TCN_1/residual_block_1/conv1D_0/kernel
2:0 2$TCN_1/residual_block_1/conv1D_0/bias
<::  2&TCN_1/residual_block_1/conv1D_1/kernel
2:0 2$TCN_1/residual_block_1/conv1D_1/bias
<::  2&TCN_1/residual_block_2/conv1D_0/kernel
2:0 2$TCN_1/residual_block_2/conv1D_0/bias
<::  2&TCN_1/residual_block_2/conv1D_1/kernel
2:0 2$TCN_1/residual_block_2/conv1D_1/bias
<::  2&TCN_1/residual_block_3/conv1D_0/kernel
2:0 2$TCN_1/residual_block_3/conv1D_0/bias
<::  2&TCN_1/residual_block_3/conv1D_1/kernel
2:0 2$TCN_1/residual_block_3/conv1D_1/bias
ќ
∆trace_02ѓ
__inference_loss_fn_0_15540П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z∆trace_0
ќ
«trace_02ѓ
__inference_loss_fn_1_15549П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z«trace_0
ќ
»trace_02ѓ
__inference_loss_fn_2_15558П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z»trace_0
ќ
…trace_02ѓ
__inference_loss_fn_3_15567П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z…trace_0
ќ
 trace_02ѓ
__inference_loss_fn_4_15576П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z trace_0
ќ
Ћtrace_02ѓ
__inference_loss_fn_5_15585П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЋtrace_0
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
0
ћ0
Ќ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
шBх
,__inference_sequential_1_layer_call_fn_13887TCN_1_input"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
,__inference_sequential_1_layer_call_fn_14020TCN_1_input"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
,__inference_sequential_1_layer_call_fn_14342inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
,__inference_sequential_1_layer_call_fn_14395inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
УBР
G__inference_sequential_1_layer_call_and_return_conditional_losses_13453TCN_1_input"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
УBР
G__inference_sequential_1_layer_call_and_return_conditional_losses_13753TCN_1_input"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ОBЛ
G__inference_sequential_1_layer_call_and_return_conditional_losses_14660inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ОBЛ
G__inference_sequential_1_layer_call_and_return_conditional_losses_14925inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ќ
[0
ќ1
ѕ2
–3
—4
“5
”6
‘7
’8
÷9
„10
Ў11
ў12
Џ13
џ14
№15
Ё16
ё17
я18
а19
б20
в21
г22
д23
е24
ж25
з26
и27
й28
к29
л30
м31
н32
о33
п34
р35
с36
т37
у38
ф39
х40
ц41
ч42
ш43
щ44
ъ45
ы46
ь47
э48"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
о
ќ0
–1
“2
‘3
÷4
Ў5
Џ6
№7
ё8
а9
в10
д11
ж12
и13
к14
м15
о16
р17
т18
ф19
ц20
ш21
ъ22
ь23"
trackable_list_wrapper
о
ѕ0
—1
”2
’3
„4
ў5
џ6
Ё7
я8
б9
г10
е11
з12
й13
л14
н15
п16
с17
у18
х19
ч20
щ21
ы22
э23"
trackable_list_wrapper
µ2≤ѓ
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
ƒBЅ
#__inference_signature_wrapper_12673x"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBг
%__inference_TCN_1_layer_call_fn_14966inputs"ѓ
®≤§
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkwjkwargs
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
жBг
%__inference_TCN_1_layer_call_fn_15007inputs"ѓ
®≤§
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkwjkwargs
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
БBю
@__inference_TCN_1_layer_call_and_return_conditional_losses_15227inputs"ѓ
®≤§
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkwjkwargs
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
БBю
@__inference_TCN_1_layer_call_and_return_conditional_losses_15447inputs"ѓ
®≤§
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkwjkwargs
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
J
50
61
72
83
94
:5"
trackable_list_wrapper
J
50
61
72
83
94
:5"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
юnon_trainable_variables
€layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
µ2≤ѓ
®≤§
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkwjkwargs
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
µ2≤ѓ
®≤§
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkwjkwargs
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Q
t0
u1
v2
w3
x4
y5
z6"
trackable_list_wrapper
д
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
З__call__
+И&call_and_return_all_conditional_losses

9kernel
:bias
!Й_jit_compiled_convolution_op"
_tf_keras_layer
Ђ
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses"
_tf_keras_layer
д
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses

5kernel
6bias
!Ц_jit_compiled_convolution_op"
_tf_keras_layer
Ђ
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses"
_tf_keras_layer
√
Э	variables
Юtrainable_variables
Яregularization_losses
†	keras_api
°__call__
+Ґ&call_and_return_all_conditional_losses
£_random_generator"
_tf_keras_layer
д
§	variables
•trainable_variables
¶regularization_losses
І	keras_api
®__call__
+©&call_and_return_all_conditional_losses

7kernel
8bias
!™_jit_compiled_convolution_op"
_tf_keras_layer
Ђ
Ђ	variables
ђtrainable_variables
≠regularization_losses
Ѓ	keras_api
ѓ__call__
+∞&call_and_return_all_conditional_losses"
_tf_keras_layer
√
±	variables
≤trainable_variables
≥regularization_losses
і	keras_api
µ__call__
+ґ&call_and_return_all_conditional_losses
Ј_random_generator"
_tf_keras_layer
Ђ
Є	variables
єtrainable_variables
Їregularization_losses
ї	keras_api
Љ__call__
+љ&call_and_return_all_conditional_losses"
_tf_keras_layer
<
;0
<1
=2
>3"
trackable_list_wrapper
<
;0
<1
=2
>3"
trackable_list_wrapper
 "
trackable_list_wrapper
і
Њnon_trainable_variables
њlayers
јmetrics
 Ѕlayer_regularization_losses
¬layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
µ2≤ѓ
®≤§
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkwjkwargs
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
µ2≤ѓ
®≤§
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkwjkwargs
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
X
Д0
Е1
Ж2
З3
И4
Й5
К6"
trackable_list_wrapper
Ђ
√	variables
ƒtrainable_variables
≈regularization_losses
∆	keras_api
«__call__
+»&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
…	variables
 trainable_variables
Ћregularization_losses
ћ	keras_api
Ќ__call__
+ќ&call_and_return_all_conditional_losses"
_tf_keras_layer
д
ѕ	variables
–trainable_variables
—regularization_losses
“	keras_api
”__call__
+‘&call_and_return_all_conditional_losses

;kernel
<bias
!’_jit_compiled_convolution_op"
_tf_keras_layer
Ђ
÷	variables
„trainable_variables
Ўregularization_losses
ў	keras_api
Џ__call__
+џ&call_and_return_all_conditional_losses"
_tf_keras_layer
√
№	variables
Ёtrainable_variables
ёregularization_losses
я	keras_api
а__call__
+б&call_and_return_all_conditional_losses
в_random_generator"
_tf_keras_layer
д
г	variables
дtrainable_variables
еregularization_losses
ж	keras_api
з__call__
+и&call_and_return_all_conditional_losses

=kernel
>bias
!й_jit_compiled_convolution_op"
_tf_keras_layer
Ђ
к	variables
лtrainable_variables
мregularization_losses
н	keras_api
о__call__
+п&call_and_return_all_conditional_losses"
_tf_keras_layer
√
р	variables
сtrainable_variables
тregularization_losses
у	keras_api
ф__call__
+х&call_and_return_all_conditional_losses
ц_random_generator"
_tf_keras_layer
Ђ
ч	variables
шtrainable_variables
щregularization_losses
ъ	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses"
_tf_keras_layer
<
?0
@1
A2
B3"
trackable_list_wrapper
<
?0
@1
A2
B3"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
эnon_trainable_variables
юlayers
€metrics
 Аlayer_regularization_losses
Бlayer_metrics
Л	variables
Мtrainable_variables
Нregularization_losses
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
µ2≤ѓ
®≤§
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkwjkwargs
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
µ2≤ѓ
®≤§
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkwjkwargs
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
X
Ф0
Х1
Ц2
Ч3
Ш4
Щ5
Ъ6"
trackable_list_wrapper
Ђ
В	variables
Гtrainable_variables
Дregularization_losses
Е	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
И	variables
Йtrainable_variables
Кregularization_losses
Л	keras_api
М__call__
+Н&call_and_return_all_conditional_losses"
_tf_keras_layer
д
О	variables
Пtrainable_variables
Рregularization_losses
С	keras_api
Т__call__
+У&call_and_return_all_conditional_losses

?kernel
@bias
!Ф_jit_compiled_convolution_op"
_tf_keras_layer
Ђ
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses"
_tf_keras_layer
√
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+†&call_and_return_all_conditional_losses
°_random_generator"
_tf_keras_layer
д
Ґ	variables
£trainable_variables
§regularization_losses
•	keras_api
¶__call__
+І&call_and_return_all_conditional_losses

Akernel
Bbias
!®_jit_compiled_convolution_op"
_tf_keras_layer
Ђ
©	variables
™trainable_variables
Ђregularization_losses
ђ	keras_api
≠__call__
+Ѓ&call_and_return_all_conditional_losses"
_tf_keras_layer
√
ѓ	variables
∞trainable_variables
±regularization_losses
≤	keras_api
≥__call__
+і&call_and_return_all_conditional_losses
µ_random_generator"
_tf_keras_layer
Ђ
ґ	variables
Јtrainable_variables
Єregularization_losses
є	keras_api
Ї__call__
+ї&call_and_return_all_conditional_losses"
_tf_keras_layer
<
C0
D1
E2
F3"
trackable_list_wrapper
<
C0
D1
E2
F3"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Љnon_trainable_variables
љlayers
Њmetrics
 њlayer_regularization_losses
јlayer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses"
_generic_user_object
µ2≤ѓ
®≤§
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkwjkwargs
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
µ2≤ѓ
®≤§
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkwjkwargs
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
X
§0
•1
¶2
І3
®4
©5
™6"
trackable_list_wrapper
Ђ
Ѕ	variables
¬trainable_variables
√regularization_losses
ƒ	keras_api
≈__call__
+∆&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
«	variables
»trainable_variables
…regularization_losses
 	keras_api
Ћ__call__
+ћ&call_and_return_all_conditional_losses"
_tf_keras_layer
д
Ќ	variables
ќtrainable_variables
ѕregularization_losses
–	keras_api
—__call__
+“&call_and_return_all_conditional_losses

Ckernel
Dbias
!”_jit_compiled_convolution_op"
_tf_keras_layer
Ђ
‘	variables
’trainable_variables
÷regularization_losses
„	keras_api
Ў__call__
+ў&call_and_return_all_conditional_losses"
_tf_keras_layer
√
Џ	variables
џtrainable_variables
№regularization_losses
Ё	keras_api
ё__call__
+я&call_and_return_all_conditional_losses
а_random_generator"
_tf_keras_layer
д
б	variables
вtrainable_variables
гregularization_losses
д	keras_api
е__call__
+ж&call_and_return_all_conditional_losses

Ekernel
Fbias
!з_jit_compiled_convolution_op"
_tf_keras_layer
Ђ
и	variables
йtrainable_variables
кregularization_losses
л	keras_api
м__call__
+н&call_and_return_all_conditional_losses"
_tf_keras_layer
√
о	variables
пtrainable_variables
рregularization_losses
с	keras_api
т__call__
+у&call_and_return_all_conditional_losses
ф_random_generator"
_tf_keras_layer
Ђ
х	variables
цtrainable_variables
чregularization_losses
ш	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
€layer_metrics
Ђ	variables
ђtrainable_variables
≠regularization_losses
ѓ__call__
+∞&call_and_return_all_conditional_losses
'∞"call_and_return_conditional_losses"
_generic_user_object
ї2Єµ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsҐ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ї2Єµ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsҐ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_dict_wrapper
—Bќ
'__inference_Dense_1_layer_call_fn_15456inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
мBй
B__inference_Dense_1_layer_call_and_return_conditional_losses_15475inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_dict_wrapper
—Bќ
'__inference_Dense_2_layer_call_fn_15484inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
мBй
B__inference_Dense_2_layer_call_and_return_conditional_losses_15503inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_dict_wrapper
–BЌ
&__inference_Output_layer_call_fn_15512inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
лBи
A__inference_Output_layer_call_and_return_conditional_losses_15531inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
≤Bѓ
__inference_loss_fn_0_15540"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≤Bѓ
__inference_loss_fn_1_15549"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≤Bѓ
__inference_loss_fn_2_15558"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≤Bѓ
__inference_loss_fn_3_15567"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≤Bѓ
__inference_loss_fn_4_15576"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≤Bѓ
__inference_loss_fn_5_15585"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
R
А	variables
Б	keras_api

Вtotal

Гcount"
_tf_keras_metric
c
Д	variables
Е	keras_api

Жtotal

Зcount
И
_fn_kwargs"
_tf_keras_metric
A:? 2-Adam/m/TCN_1/residual_block_0/conv1D_0/kernel
A:? 2-Adam/v/TCN_1/residual_block_0/conv1D_0/kernel
7:5 2+Adam/m/TCN_1/residual_block_0/conv1D_0/bias
7:5 2+Adam/v/TCN_1/residual_block_0/conv1D_0/bias
A:?  2-Adam/m/TCN_1/residual_block_0/conv1D_1/kernel
A:?  2-Adam/v/TCN_1/residual_block_0/conv1D_1/kernel
7:5 2+Adam/m/TCN_1/residual_block_0/conv1D_1/bias
7:5 2+Adam/v/TCN_1/residual_block_0/conv1D_1/bias
H:F 24Adam/m/TCN_1/residual_block_0/matching_conv1D/kernel
H:F 24Adam/v/TCN_1/residual_block_0/matching_conv1D/kernel
>:< 22Adam/m/TCN_1/residual_block_0/matching_conv1D/bias
>:< 22Adam/v/TCN_1/residual_block_0/matching_conv1D/bias
A:?  2-Adam/m/TCN_1/residual_block_1/conv1D_0/kernel
A:?  2-Adam/v/TCN_1/residual_block_1/conv1D_0/kernel
7:5 2+Adam/m/TCN_1/residual_block_1/conv1D_0/bias
7:5 2+Adam/v/TCN_1/residual_block_1/conv1D_0/bias
A:?  2-Adam/m/TCN_1/residual_block_1/conv1D_1/kernel
A:?  2-Adam/v/TCN_1/residual_block_1/conv1D_1/kernel
7:5 2+Adam/m/TCN_1/residual_block_1/conv1D_1/bias
7:5 2+Adam/v/TCN_1/residual_block_1/conv1D_1/bias
A:?  2-Adam/m/TCN_1/residual_block_2/conv1D_0/kernel
A:?  2-Adam/v/TCN_1/residual_block_2/conv1D_0/kernel
7:5 2+Adam/m/TCN_1/residual_block_2/conv1D_0/bias
7:5 2+Adam/v/TCN_1/residual_block_2/conv1D_0/bias
A:?  2-Adam/m/TCN_1/residual_block_2/conv1D_1/kernel
A:?  2-Adam/v/TCN_1/residual_block_2/conv1D_1/kernel
7:5 2+Adam/m/TCN_1/residual_block_2/conv1D_1/bias
7:5 2+Adam/v/TCN_1/residual_block_2/conv1D_1/bias
A:?  2-Adam/m/TCN_1/residual_block_3/conv1D_0/kernel
A:?  2-Adam/v/TCN_1/residual_block_3/conv1D_0/kernel
7:5 2+Adam/m/TCN_1/residual_block_3/conv1D_0/bias
7:5 2+Adam/v/TCN_1/residual_block_3/conv1D_0/bias
A:?  2-Adam/m/TCN_1/residual_block_3/conv1D_1/kernel
A:?  2-Adam/v/TCN_1/residual_block_3/conv1D_1/kernel
7:5 2+Adam/m/TCN_1/residual_block_3/conv1D_1/bias
7:5 2+Adam/v/TCN_1/residual_block_3/conv1D_1/bias
%:#  2Adam/m/Dense_1/kernel
%:#  2Adam/v/Dense_1/kernel
: 2Adam/m/Dense_1/bias
: 2Adam/v/Dense_1/bias
%:#  2Adam/m/Dense_2/kernel
%:#  2Adam/v/Dense_2/kernel
: 2Adam/m/Dense_2/bias
: 2Adam/v/Dense_2/bias
$:" 2Adam/m/Output/kernel
$:" 2Adam/v/Output/kernel
:2Adam/m/Output/bias
:2Adam/v/Output/bias
 "
trackable_list_wrapper
_
t0
u1
v2
w3
x4
y5
z6
r7
s8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Эnon_trainable_variables
Юlayers
Яmetrics
 †layer_regularization_losses
°layer_metrics
Э	variables
Юtrainable_variables
Яregularization_losses
°__call__
+Ґ&call_and_return_all_conditional_losses
'Ґ"call_and_return_conditional_losses"
_generic_user_object
њ
Ґtrace_0
£trace_12Д
*__inference_SDropout_0_layer_call_fn_15590
*__inference_SDropout_0_layer_call_fn_15595©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zҐtrace_0z£trace_1
х
§trace_0
•trace_12Ї
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15600
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15605©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z§trace_0z•trace_1
"
_generic_user_object
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
¶non_trainable_variables
Іlayers
®metrics
 ©layer_regularization_losses
™layer_metrics
§	variables
•trainable_variables
¶regularization_losses
®__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ђnon_trainable_variables
ђlayers
≠metrics
 Ѓlayer_regularization_losses
ѓlayer_metrics
Ђ	variables
ђtrainable_variables
≠regularization_losses
ѓ__call__
+∞&call_and_return_all_conditional_losses
'∞"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
∞non_trainable_variables
±layers
≤metrics
 ≥layer_regularization_losses
іlayer_metrics
±	variables
≤trainable_variables
≥regularization_losses
µ__call__
+ґ&call_and_return_all_conditional_losses
'ґ"call_and_return_conditional_losses"
_generic_user_object
њ
µtrace_0
ґtrace_12Д
*__inference_SDropout_1_layer_call_fn_15610
*__inference_SDropout_1_layer_call_fn_15615©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zµtrace_0zґtrace_1
х
Јtrace_0
Єtrace_12Ї
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15620
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15625©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЈtrace_0zЄtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
єnon_trainable_variables
Їlayers
їmetrics
 Љlayer_regularization_losses
љlayer_metrics
Є	variables
єtrainable_variables
Їregularization_losses
Љ__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
h
Д0
Е1
Ж2
З3
И4
Й5
К6
В7
Г8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Њnon_trainable_variables
њlayers
јmetrics
 Ѕlayer_regularization_losses
¬layer_metrics
√	variables
ƒtrainable_variables
≈regularization_losses
«__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
ї2Єµ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsҐ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ї2Єµ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsҐ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
√non_trainable_variables
ƒlayers
≈metrics
 ∆layer_regularization_losses
«layer_metrics
…	variables
 trainable_variables
Ћregularization_losses
Ќ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
ѕ	variables
–trainable_variables
—regularization_losses
”__call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ќnon_trainable_variables
ќlayers
ѕmetrics
 –layer_regularization_losses
—layer_metrics
÷	variables
„trainable_variables
Ўregularization_losses
Џ__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
№	variables
Ёtrainable_variables
ёregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
њ
„trace_0
Ўtrace_12Д
*__inference_SDropout_0_layer_call_fn_15630
*__inference_SDropout_0_layer_call_fn_15635©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z„trace_0zЎtrace_1
х
ўtrace_0
Џtrace_12Ї
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15640
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15645©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zўtrace_0zЏtrace_1
"
_generic_user_object
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
џnon_trainable_variables
№layers
Ёmetrics
 ёlayer_regularization_losses
яlayer_metrics
г	variables
дtrainable_variables
еregularization_losses
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
аnon_trainable_variables
бlayers
вmetrics
 гlayer_regularization_losses
дlayer_metrics
к	variables
лtrainable_variables
мregularization_losses
о__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
р	variables
сtrainable_variables
тregularization_losses
ф__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
њ
кtrace_0
лtrace_12Д
*__inference_SDropout_1_layer_call_fn_15650
*__inference_SDropout_1_layer_call_fn_15655©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zкtrace_0zлtrace_1
х
мtrace_0
нtrace_12Ї
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15660
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15665©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zмtrace_0zнtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
оnon_trainable_variables
пlayers
рmetrics
 сlayer_regularization_losses
тlayer_metrics
ч	variables
шtrainable_variables
щregularization_losses
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
h
Ф0
Х1
Ц2
Ч3
Ш4
Щ5
Ъ6
Т7
У8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
В	variables
Гtrainable_variables
Дregularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
ї2Єµ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsҐ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ї2Єµ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsҐ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
И	variables
Йtrainable_variables
Кregularization_losses
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
эnon_trainable_variables
юlayers
€metrics
 Аlayer_regularization_losses
Бlayer_metrics
О	variables
Пtrainable_variables
Рregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses"
_generic_user_object
њ
Мtrace_0
Нtrace_12Д
*__inference_SDropout_0_layer_call_fn_15670
*__inference_SDropout_0_layer_call_fn_15675©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zМtrace_0zНtrace_1
х
Оtrace_0
Пtrace_12Ї
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15680
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15685©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zОtrace_0zПtrace_1
"
_generic_user_object
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
Ґ	variables
£trainable_variables
§regularization_losses
¶__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
©	variables
™trainable_variables
Ђregularization_losses
≠__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
ѓ	variables
∞trainable_variables
±regularization_losses
≥__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
њ
Яtrace_0
†trace_12Д
*__inference_SDropout_1_layer_call_fn_15690
*__inference_SDropout_1_layer_call_fn_15695©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЯtrace_0z†trace_1
х
°trace_0
Ґtrace_12Ї
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15700
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15705©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z°trace_0zҐtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
£non_trainable_variables
§layers
•metrics
 ¶layer_regularization_losses
Іlayer_metrics
ґ	variables
Јtrainable_variables
Єregularization_losses
Ї__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
h
§0
•1
¶2
І3
®4
©5
™6
Ґ7
£8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
®non_trainable_variables
©layers
™metrics
 Ђlayer_regularization_losses
ђlayer_metrics
Ѕ	variables
¬trainable_variables
√regularization_losses
≈__call__
+∆&call_and_return_all_conditional_losses
'∆"call_and_return_conditional_losses"
_generic_user_object
ї2Єµ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsҐ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ї2Єµ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsҐ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
≠non_trainable_variables
Ѓlayers
ѓmetrics
 ∞layer_regularization_losses
±layer_metrics
«	variables
»trainable_variables
…regularization_losses
Ћ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
≤non_trainable_variables
≥layers
іmetrics
 µlayer_regularization_losses
ґlayer_metrics
Ќ	variables
ќtrainable_variables
ѕregularization_losses
—__call__
+“&call_and_return_all_conditional_losses
'“"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Јnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
‘	variables
’trainable_variables
÷regularization_losses
Ў__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Љnon_trainable_variables
љlayers
Њmetrics
 њlayer_regularization_losses
јlayer_metrics
Џ	variables
џtrainable_variables
№regularization_losses
ё__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
њ
Ѕtrace_0
¬trace_12Д
*__inference_SDropout_0_layer_call_fn_15710
*__inference_SDropout_0_layer_call_fn_15715©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЅtrace_0z¬trace_1
х
√trace_0
ƒtrace_12Ї
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15720
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15725©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z√trace_0zƒtrace_1
"
_generic_user_object
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
≈non_trainable_variables
∆layers
«metrics
 »layer_regularization_losses
…layer_metrics
б	variables
вtrainable_variables
гregularization_losses
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 non_trainable_variables
Ћlayers
ћmetrics
 Ќlayer_regularization_losses
ќlayer_metrics
и	variables
йtrainable_variables
кregularization_losses
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ѕnon_trainable_variables
–layers
—metrics
 “layer_regularization_losses
”layer_metrics
о	variables
пtrainable_variables
рregularization_losses
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
њ
‘trace_0
’trace_12Д
*__inference_SDropout_1_layer_call_fn_15730
*__inference_SDropout_1_layer_call_fn_15735©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z‘trace_0z’trace_1
х
÷trace_0
„trace_12Ї
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15740
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15745©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z÷trace_0z„trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ўnon_trainable_variables
ўlayers
Џmetrics
 џlayer_regularization_losses
№layer_metrics
х	variables
цtrainable_variables
чregularization_losses
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ю2ЫШ
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
0
В0
Г1"
trackable_list_wrapper
.
А	variables"
_generic_user_object
:  (2total
:  (2count
0
Ж0
З1"
trackable_list_wrapper
.
Д	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
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
еBв
*__inference_SDropout_0_layer_call_fn_15590inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
еBв
*__inference_SDropout_0_layer_call_fn_15595inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15600inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15605inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
еBв
*__inference_SDropout_1_layer_call_fn_15610inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
еBв
*__inference_SDropout_1_layer_call_fn_15615inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15620inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15625inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
еBв
*__inference_SDropout_0_layer_call_fn_15630inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
еBв
*__inference_SDropout_0_layer_call_fn_15635inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15640inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15645inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
еBв
*__inference_SDropout_1_layer_call_fn_15650inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
еBв
*__inference_SDropout_1_layer_call_fn_15655inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15660inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15665inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
еBв
*__inference_SDropout_0_layer_call_fn_15670inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
еBв
*__inference_SDropout_0_layer_call_fn_15675inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15680inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15685inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
еBв
*__inference_SDropout_1_layer_call_fn_15690inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
еBв
*__inference_SDropout_1_layer_call_fn_15695inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15700inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15705inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
еBв
*__inference_SDropout_0_layer_call_fn_15710inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
еBв
*__inference_SDropout_0_layer_call_fn_15715inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15720inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15725inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
еBв
*__inference_SDropout_1_layer_call_fn_15730inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
еBв
*__inference_SDropout_1_layer_call_fn_15735inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15740inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15745inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
trackable_dict_wrapper©
B__inference_Dense_1_layer_call_and_return_conditional_losses_15475c#$/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ ",Ґ)
"К
tensor_0€€€€€€€€€ 
Ъ Г
'__inference_Dense_1_layer_call_fn_15456X#$/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "!К
unknown€€€€€€€€€ ©
B__inference_Dense_2_layer_call_and_return_conditional_losses_15503c+,/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ ",Ґ)
"К
tensor_0€€€€€€€€€ 
Ъ Г
'__inference_Dense_2_layer_call_fn_15484X+,/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "!К
unknown€€€€€€€€€ ®
A__inference_Output_layer_call_and_return_conditional_losses_15531c34/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ В
&__inference_Output_layer_call_fn_15512X34/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "!К
unknown€€€€€€€€€ў
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15600ПIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "BҐ?
8К5
tensor_0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ў
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15605ПIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "BҐ?
8К5
tensor_0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ў
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15640ПIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "BҐ?
8К5
tensor_0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ў
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15645ПIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "BҐ?
8К5
tensor_0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ў
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15680ПIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "BҐ?
8К5
tensor_0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ў
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15685ПIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "BҐ?
8К5
tensor_0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ў
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15720ПIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "BҐ?
8К5
tensor_0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ў
E__inference_SDropout_0_layer_call_and_return_conditional_losses_15725ПIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "BҐ?
8К5
tensor_0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≥
*__inference_SDropout_0_layer_call_fn_15590ДIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "7К4
unknown'€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
*__inference_SDropout_0_layer_call_fn_15595ДIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "7К4
unknown'€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
*__inference_SDropout_0_layer_call_fn_15630ДIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "7К4
unknown'€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
*__inference_SDropout_0_layer_call_fn_15635ДIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "7К4
unknown'€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
*__inference_SDropout_0_layer_call_fn_15670ДIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "7К4
unknown'€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
*__inference_SDropout_0_layer_call_fn_15675ДIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "7К4
unknown'€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
*__inference_SDropout_0_layer_call_fn_15710ДIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "7К4
unknown'€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
*__inference_SDropout_0_layer_call_fn_15715ДIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "7К4
unknown'€€€€€€€€€€€€€€€€€€€€€€€€€€€ў
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15620ПIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "BҐ?
8К5
tensor_0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ў
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15625ПIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "BҐ?
8К5
tensor_0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ў
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15660ПIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "BҐ?
8К5
tensor_0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ў
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15665ПIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "BҐ?
8К5
tensor_0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ў
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15700ПIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "BҐ?
8К5
tensor_0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ў
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15705ПIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "BҐ?
8К5
tensor_0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ў
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15740ПIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "BҐ?
8К5
tensor_0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ў
E__inference_SDropout_1_layer_call_and_return_conditional_losses_15745ПIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "BҐ?
8К5
tensor_0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≥
*__inference_SDropout_1_layer_call_fn_15610ДIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "7К4
unknown'€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
*__inference_SDropout_1_layer_call_fn_15615ДIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "7К4
unknown'€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
*__inference_SDropout_1_layer_call_fn_15650ДIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "7К4
unknown'€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
*__inference_SDropout_1_layer_call_fn_15655ДIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "7К4
unknown'€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
*__inference_SDropout_1_layer_call_fn_15690ДIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "7К4
unknown'€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
*__inference_SDropout_1_layer_call_fn_15695ДIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "7К4
unknown'€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
*__inference_SDropout_1_layer_call_fn_15730ДIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "7К4
unknown'€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
*__inference_SDropout_1_layer_call_fn_15735ДIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "7К4
unknown'€€€€€€€€€€€€€€€€€€€€€€€€€€€ј
@__inference_TCN_1_layer_call_and_return_conditional_losses_15227|56789:;<=>?@ABCDEF8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€В
p
™ ",Ґ)
"К
tensor_0€€€€€€€€€ 
Ъ ј
@__inference_TCN_1_layer_call_and_return_conditional_losses_15447|56789:;<=>?@ABCDEF8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€В
p 
™ ",Ґ)
"К
tensor_0€€€€€€€€€ 
Ъ Ъ
%__inference_TCN_1_layer_call_fn_14966q56789:;<=>?@ABCDEF8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€В
p
™ "!К
unknown€€€€€€€€€ Ъ
%__inference_TCN_1_layer_call_fn_15007q56789:;<=>?@ABCDEF8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€В
p 
™ "!К
unknown€€€€€€€€€ Ђ
 __inference__wrapped_model_12915Ж56789:;<=>?@ABCDEF#$+,349Ґ6
/Ґ,
*К'
TCN_1_input€€€€€€€€€В
™ "/™,
*
Output К
output€€€€€€€€€C
__inference_loss_fn_0_15540$#Ґ

Ґ 
™ "К
unknown C
__inference_loss_fn_1_15549$$Ґ

Ґ 
™ "К
unknown C
__inference_loss_fn_2_15558$+Ґ

Ґ 
™ "К
unknown C
__inference_loss_fn_3_15567$,Ґ

Ґ 
™ "К
unknown C
__inference_loss_fn_4_15576$3Ґ

Ґ 
™ "К
unknown C
__inference_loss_fn_5_15585$4Ґ

Ґ 
™ "К
unknown „
G__inference_sequential_1_layer_call_and_return_conditional_losses_13453Л56789:;<=>?@ABCDEF#$+,34AҐ>
7Ґ4
*К'
TCN_1_input€€€€€€€€€В
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ „
G__inference_sequential_1_layer_call_and_return_conditional_losses_13753Л56789:;<=>?@ABCDEF#$+,34AҐ>
7Ґ4
*К'
TCN_1_input€€€€€€€€€В
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ “
G__inference_sequential_1_layer_call_and_return_conditional_losses_14660Ж56789:;<=>?@ABCDEF#$+,34<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€В
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ “
G__inference_sequential_1_layer_call_and_return_conditional_losses_14925Ж56789:;<=>?@ABCDEF#$+,34<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€В
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ ±
,__inference_sequential_1_layer_call_fn_13887А56789:;<=>?@ABCDEF#$+,34AҐ>
7Ґ4
*К'
TCN_1_input€€€€€€€€€В
p

 
™ "!К
unknown€€€€€€€€€±
,__inference_sequential_1_layer_call_fn_14020А56789:;<=>?@ABCDEF#$+,34AҐ>
7Ґ4
*К'
TCN_1_input€€€€€€€€€В
p 

 
™ "!К
unknown€€€€€€€€€Ђ
,__inference_sequential_1_layer_call_fn_14342{56789:;<=>?@ABCDEF#$+,34<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€В
p

 
™ "!К
unknown€€€€€€€€€Ђ
,__inference_sequential_1_layer_call_fn_14395{56789:;<=>?@ABCDEF#$+,34<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€В
p 

 
™ "!К
unknown€€€€€€€€€Ъ
#__inference_signature_wrapper_12673s56789:;<=>?@ABCDEF#$+,34+Ґ(
Ґ 
!™

xК
xВ"*™'
%
output_0К
output_0