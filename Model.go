package main
import (
	"github.com/kr/pretty"
	"math"
	"os"
	"io/ioutil"
)

type Model struct{
	Layers []Layer
}


func NewModel(hidden_layers...int)Model{
	if (len(hidden_layers) < 2) {
		return Model{nil}
	}
	m := Model{}
	for i := 0; i < len(hidden_layers)-1; i++ {
		m.NewLayer(hidden_layers[i],i,"relu")
	}
	m.NewLayer(hidden_layers[len(hidden_layers)-1],len(hidden_layers)-1,"sig");
	return m

}

func(m *Model)Predict(x []float64)[]float64{
	if(len(x) != m.Layers[0].Uints){
		print("Input Size is not expected\n")
		return []float64{}
	}
	for i := 0; i < len(m.Layers); i++ {
		pretty.Println(i,x)
		x,_ = m.Layers[i].Compute(x)
	}
	return x
}


type Layer struct{
	Uints int
	Weights []float64
	Biases []float64
	Position int
	Model *Model
	Activation func(z []float64)[]float64
}

//a1 = w1*x1 + b

func(m *Model) NewLayer(uints,n int,act string){
	layer := Layer{uints,[]float64{0},[]float64{0},n,m,nil}
	for i := 1; i < uints;i++ {
		layer.Weights = append(layer.Weights,0);
		layer.Biases = append(layer.Biases,0);
	}
	if(act == "sig"){
		layer.Activation = Sigmoid
	}else{
		layer.Activation = Relu
	}
	m.Layers = append(m.Layers,layer)
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
func ReadCSV(fileName string){
	file, err := os.Open(fileName)
	if(err != nil){
		panic(err)
	}
	data, err := ioutil.ReadAll(file)
	if(err != nil){
		panic(err)
	}
	
	pretty.Println(data)
	return 
}
/*
func ReadMNIST(fileName string){
	file, err := os.Open(fileName)
	if(err != nil){
		panic(err)
	}
	var data = [][]float64{
		[]float64{0},
	}
	var buf = make([]byte,256)
	var n = 1
	var row = []float64{0}
	for n != 0 {
		n, err := file.Read(buf);
		for i := 0; i < len(buf); i++ {
			if(buf[i] == '\n'){
				row[]
			}
		}
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
	data = [][]float64{datum}
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
				if(n_row > 1){
					data[n_row] = datum
				}
				n_row +=1;
			}
		}

		//pretty.Println("Total Bytes",bytes_read,n);
		
		bytes_read += n
	}
	pretty.Println(data)
	return data

}
*/


func main(){
	ReadCSV("./test.csv")
	//x_train,x_test,y_train,ytest = 
	model := NewModel(3,19)
	pretty.Println(model)
	pretty.Println(model.Predict([]float64{0.2,1.2,1.2}))
	//pretty.Println(model.Learn())
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