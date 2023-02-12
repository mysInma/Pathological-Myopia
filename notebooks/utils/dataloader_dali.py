import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali as dali

#Defino una clase para dataloader

class SimpleDALIGenericIterator(dali.pipeline.Pipeline):
    def __init__(self, image_dir, batch_size, device_id, num_threads=2): #device_id= is del dispositivo en el que se ejecutará iteración (ej: GPU/CPU)
        super(SimpleDALIGenericIterator, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id) #para invocar al constructor de la clase,             superClass, para inicializar los atributos que se heredan de la clase base.
        self.input = ops.FileReader(file_root=image_dir, random_shuffle=True) #leer los archivos de la imagen.
    
    #Para definir el flujo de trabajo para leer archivos JPEG y decodificarlos en imágenes RGB:
    def define_graph(self): #grafo computacional para describir el modelo y realizar los cálculos necesarios para el entrenamiento y la inferencia.
        jpegs, labels = self.input(name="Reader") #leer jpeg y las estiquetas desde el origen de archivos.
        images = ops.ImageDecoder(device="mixed", output_type=types.RGB)(jpegs) #los archivos JPEG son decodififcados en las imágenes RGB
        return [images]
    
if __name__ == "__main__":
    
    train_pipe = SimpleDALIGenericIterator("../test", 6, 0)
    train_pipe.build() #Se construye el pipeline
    
    for i in range(10): #bucle de 10 iteraciones, para obtener una tanda de datos de 6 imágenes
        data = train_pipe.run() #contiene los tenserores DALI con las imágenes decodificadas.
        print(data)
        images = data[0].as_cpu().as_array() #se accede primera entra de la lista y se convierte en una matriz numpy
        
    
    
    
    

