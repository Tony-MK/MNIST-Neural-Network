package main
import (
	"github.com/kr/pretty"
	"math"
	//"strconv"
)

type Model struct{
	Layers []Layer
	NLayers int
}


func NewModel(hidden_layers...int)Model{
	m := Model{}
	if (len(hidden_layers) > 2) {
		for i := 0; i < len(hidden_layers)-1; i++ {
			m.NewLayer(hidden_layers[i],i,"relu")
		}
	}
	m.NewLayer(hidden_layers[len(hidden_layers)-1],len(hidden_layers)-1,"sig");
	return m
}

func(m *Model)Predict(x []float64)[]float64{
	if(len(x) != m.Layers[0].Uints){
		print("Input Size is not expected\n")
		return []float64{}
	}
	return m.Think(x,0)
}
func(m *Model)Think(x []float64,n int)[]float64{
	_,a := m.Layers[n].Compute(x)
	if(1 == m.NLayers-n){
		return a
	}
	return m.Think(a,n+1)
}

func(m *Model)Learn(xTrain [][]float64,yTrain [][]float64)float64{
	// dC/dW = dC/dA * dA/dZ * dZ/dW
	var cost = make([]float64,m.Layers[m.NLayers-1].Uints)
	//var delta = [][]float64{0}
	for i,x := range(xTrain){
		if(len(yTrain[i]) != m.Layers[m.NLayers-1].Uints){
			panic("expected Shape")
		}
		var Zvalues = [][]float64{}
		var Avalues = [][]float64{}
		for i := 0; i < m.NLayers; i++ {
			z,a := m.Layers[m.NLayers-1].Compute(x);
			Zvalues= append(Zvalues,z);Avalues= append(Avalues,a)
		}
		for n := 0; n < m.Layers[m.NLayers-1].Uints; n++  {
			cost[n] += Avalues[m.NLayers-1][n]-yTrain[i][n]
		}
		pretty.Println(cost)
	}
	var average_cost  = 0.0

	for i := 0; i < m.Layers[m.NLayers-1].Uints; i++ {
		average_cost += cost[i]
	}
	average_cost /= float64(m.Layers[m.NLayers-1].Uints)
	return average_cost

}

type Layer struct{
	Uints int
	Weights []float64
	Biases []float64
	Position int
	Model *Model
	Activation func(z []float64)[]float64
}
func(m *Model)NewLayer(uints,n int,act string){

	layer := Layer{uints,make([]float64,uints),make([]float64,uints),n,m,nil}

	if(act == "sig"){
		layer.Activation = Sigmoid
	}else{
		layer.Activation = Relu
	}
	m.Layers = append(m.Layers,layer)
	m.NLayers = len(m.Layers);
	return
}



// (1,3) X (1,4) = 1,4
func(l *Layer)Compute(x []float64)(z []float64,a []float64){
	z = make([]float64,l.Uints)
	for n_neuron := 0; n_neuron < l.Uints; n_neuron++ {
		for n_input := 0; n_input < len(x); n_input++ {
			z[n_neuron] += x[n_input]*l.Weights[n_neuron]
		}
		z[n_neuron] += l.Biases[n_neuron]
	}
	return z,l.Activation(z)
}


func CheckError(err error){
	if(err != nil){
		panic(err)
	}
}
/*
const N_PIXELS = 784
const Row_Delimeter = 0x5c6e
const Col_Delimeter = 0x2c
func ReadMNIST(fileName string)(data [][]float64){
	file, err := os.Open(fileName)
	if(err != nil){
		panic(err)
		return data
	}
	var n = -1
	var buf = make([]byte,256);
	var bytes_read = 0;
	datum := make([]float64,N_PIXELS);
	data = [][]float64{}
	var n_row = 0
	var n_col = 0;
	var pi = 0
	for (n != 0){
		n, err = file.Read(buf)
		if(err != nil){
			break
		}
		for i := 0; i < len(buf); i++ {
			if(buf[i] == Col_Delimeter){
				for sFig := 0; sFig < i-pi; sFig++ {
					datum[n_col] += float64(buf[pi+sFig])
					for p := 0; p < sFig; p++ {datum[n_col] *= 10}
				}
				n_col += 1;pi = i;
			}

			if(buf[i] == '\n'){
				n_col = 0;
				data = append(data,datum)
				n_row +=1;
			}
		}


		//pretty.Println("Total Bytes",bytes_read,data);
		
		bytes_read += n
	}
	pretty.Println(data)
	return data

}
*/

var XData = [][]float64{
	{1.0,1.0,1.0},
	{1.0,0.0,1.0},
	{1.0,0.0,0.0},
	{0.0,0.0,0.0},
}
var YData = [][]float64{
	{1.0,0.0},
	{1.0,0.0},
	{0.0,1.0},
	{0.0,1.0},
}

func main(){
	///ReadMNIST("./test.csv")
	//x_train,x_test,y_train,ytest = 
	model := NewModel(2)
	pretty.Println(model)
	pretty.Println(model.Predict([]float64{0.2}))
	pretty.Println(model.Learn(XData,YData))
}


func Relu(z []float64)[]float64{
	for i := 0; i < len(z); i++ {
		if(z[i] < 0){
			z[i] = 0
		}
		
	}
	return z
}

func Sigmoid(z []float64)[]float64{
	for i := 0; i < len(z); i++ {
		z[i] = math.Exp(z[i])
	}
	return z
}