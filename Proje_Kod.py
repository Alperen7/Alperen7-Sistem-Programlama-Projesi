import warnings
warnings.filterwarnings("ignore") #normalde hata  olmayan ama bazı konularda bize uyaran her şeyi kapatıyor
#hataları değil, uyarıları görmemek için kapattım. Kullanılmaya da bilir.

from PyQt5.QtWidgets import * #tüm PyQt5 bileşenlerini yükler

import matplotlib.pyplot as plt  #veri görselleştirmesinde kullanılır. 2 ve 3 boyutlu çizimler yapmayı sağlar.
import numpy as np #yardımcı kütüphanedir. Çok boyutlu dizilerle, matrislerle yapılan matematiksel işlemlerde kullanılır.
import pandas as pd #yardımcı kütüphanedir. Veri işleme ve analiz için yazılmış bir yazılım kütüphanesidir
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas #çizim kütüphanesidir.
from matplotlib.figure import Figure  # yardımcı kütüphanedir. Verileri görselleştirmeye yarar.

class Window(QMainWindow): #Sınıf işlemi için QMain'de inherad etme işlemi
    
    def __init__(self): #constracter tanımı
        super().__init__() #Qwidget metotlarını kullanabilirm demektir.
        
        self.left = 50 #pencerenin nereden açılacağını belirledim
        self.top = 50
        self.width = 1080 #pencerenin boyutlarını belirledim
        self.height = 640
        self.title = "Kümeleme İşlemi Kullanıcı Arayüzü" #pencerenin başlığı
        
        self.setWindowTitle(self.title) #yukarıdaki parametrelerin set işlemi
        self.setGeometry(self.left, self.top ,self.width, self.height ) #yukarıdaki parametrelerin set işlemi
        
        self.k = 1 #seçilecek K değerini default olarak 1 ile başlatıyor
        self.save_txt = "" #kayıt tutabilmek için boş bir değişken tanımladım
                
        self.tabWidget() #tab'ı pencere üzerinde görüntülemeye yarayan metot
        self.widgets() #tüm yazılan bileşenleri pencere üzerinde görüntülemeye yarayan metot
        self.layouts() #layaot'ları pencere üzerinde görüntülemeye yarayan metot
        self.prepareData() #veri setini görselleştirme işlemi yapan fonksiyonu görüntülemeye yarayan metot
        
        self.show()    #oluşturulan pencereyi görselleştirmeye yarar
        
    def tabWidget(self):
        self.tabs = QTabWidget() #tab oluşturuldu
        self.setCentralWidget(self.tabs)
        self.tab1 = QWidget() #tab'ın altında 1 tab tanımlandı
        self.tabs.addTab(self.tab1,"Menü") #tab'ın adı belirlendi
        
    def widgets(self): #bahsettiğim bileşenlerin kodlaması   
    
        # plot
        self.p = PlotCanvas(self,width = 5, height = 5)
    
        # label K Değeri
        self.k_number_text = QLabel("K Değerini Belirle:") #başında self olursa sadece label için bir variable olmaktan çıkar 
                                                                                       #window için bir variable konumuna gelir.     
        # spin box
        self.k_number = QSpinBox(self)
        self.k_number.setMinimum(1) #spinbox'ın min değeri
        self.k_number.setMaximum(9) #spinbox'ın max değeri, setRange ile tek satırda tanımlamakta mümkün
        self.k_number.setSingleStep(1) #1'den 9'a birer birer gidecek diyor
        self.k_number.valueChanged.connect(self.k_numberFunction)
        
        # radio button
        self.text_save = QRadioButton("Metni kaydet",self) #1.radio buton oluşturuldu
        self.plot_save = QRadioButton("Plot  Kaydet",self) #2.radio buton oluşturuldu
        self.text_plot_save = QRadioButton("Metni ve Plot'u Kaydet",self) #3.radio buton oluşturuldu
        self.text_plot_save.setChecked(True) #ekranda seçili gelen default parametresi tanımlandı
        
        #Kümeleme Butonu
        self.cluster = QPushButton("Kümeleme Yap",self)
        self.cluster.clicked.connect(self.clusterFunction)
    
        # list
        self.result_list = QListWidget(self) #listWidget oluşturuldu
    
    def prepareData(self):
        
        self.p.clear() #eğer plot'ta bir veri varsa önce temizler
        
        data = pd.read_csv("veri_seti.csv") #veri setini okuma
        
        self.f1 = data.iloc[:,3].values #tablonun 3.sütununu alıyor
        self.f2 = data.iloc[:,4].values #tablonun 4.sütununu alıyor
        
        X = np.array(list(zip(self.f1,self.f2))) #yukarıdaki sütunları birleştirerek liste haline getiriyor
        
        self.C_x = np.random.randint(0, np.max(X) - 20, size = self.k) #plot'un üstünde k sayısı kadar random şekilde centroid oluşturur, x ekseni
        self.C_y = np.random.randint(0, np.max(X) - 20, size = self.k) #plot'un üstünde k sayısı kadar random şekilde centroid oluşturur, y ekseni
        
        self.p.plot(self.f1,self.f2,"black",7)  #f1 ve f2 veri setini görselleştiriyor, siyah renkte, boyut 7
        self.p.plot(self.C_x,self.C_y,"red",200,"*") #gruplama yapmak için centroid'i oluşturdum. renk - boyut - şekil
    
    def k_numberFunction(self):
        self.k = self.k_number.value() #spin box'a girilen değeri alır, bu değeri kümelemede kullanıcam
        self.prepareData() #k değeri ile plot'u da değiştiren metot oluşturudum
    
    def dist(self,a,b): #a ve b input parametreli distance fonksiyonu
        return np.linalg.norm(a - b, axis = 1) #a-b arasındaki mesafeyi, O.D.'ı bulur.
    
    def kMeansClustering(self,f1,f2,C_x,C_y,k): #K-Means Kümeleme metodu. İnput parametresi olarak bu değerleri aldı
        
        X = np.array(list(zip(f1,f2))) #yıllık gelir ile harcama puanı zip'lendi
        
        C = np.array(list(zip(C_x,C_y))) #x ve centroid'leri zip'lendi.
        #np.array kütüpjanesini kıllandım çünkü aşağıda matematiksel işlemlere tabii olacak
        
        clusters = np.zeros(len(X)) #Sıfırlardan oluşan ve veri uzunluğunda olan dizi tanımlandı
        
        for z in range(10): #10 rastgele bir sayıdır. 10kez centroit'leri tanımla demektir.
            
            for i in range(len(X)): #en yakın noktaların bulunacağı döngü
                
                distances = self.dist(X[i], C) #distance(mesafe) fonksiyonu ile kaşılaştırma yapıyor. Veri ile centroid arasındaki mesafeyi bulucak. Ö.D.'ı buluyor
                cluster = np.argmin(distances) #daha kısa olan uzaklığı ve ona ait olan centroid'i bulur.
                clusters[i] = cluster #bulduğu veriyi diziye depolar.
                
            for i in range(k): #yeniden kümeleme ve yeni merkez oluşturma işlemi
                
                points = [ X[j] for j in range(len(X)) if clusters[j] == i] #k'lar gruplandı
                C[i] = np.mean(points, axis = 0) #seçili grubun ortalamasını alır ve yeni merkezi bulma işlemini yapar
        colors = ['black', 'red', 'blue','yellow', 'cyan', 'magenta',"darkgreen","silver","indigo","maroon"]
        #yapılan bu işlem 10 kez tekrarlanacak ve oluşabilecek her yeni merkez için renklendirme işlemi
        
        for i in range(k): #üstteki for'un görselleştirme işlemi 
            
            points = np.array([ X[j] for j in range(len(X)) if clusters[j] == i]) # her k için farklı rengi tanımlıyor
            self.p.plot(points[:,0],points[:,1],colors[i],7) #veriler için görselleştirme işlemi
            self.p.plot(C[:,0],C[:,1],"red",200,"*") #kümelerin yeni merkezlerini görselleştirme
            
            result_txt = "Kümeleme #"+str(i+1)+": "+str(len(points)) + " ("+colors[i]+")" #yularıdaki döngülerin sonuç kısmında görüntülenmesini sağlar
            self.result_list.addItems([result_txt]) #Kümele 1 : adet (Renk) şeklinde oluşturur.
            #sonucu arayüzde görselleştirmeyi sağlar
            
            self.save_txt = self.save_txt + result_txt + " -- " #başta oluşturulan boş dosyaya çıkan sonuçları yazar
            #iki çizgi karışmasın diye. Birden fazla cluster var
            
    def clusterFunction(self):
        
        self.result_list.clear() #kümeleme yap butonundan sonra sonuç bölümünü yeniler
        self.p.clear() #kümeleme yap butonundan sonra plot'u yeniler
        
        self.kMeansClustering(self.f1, self.f2, self.C_x, self.C_y, self.k) 
        #kazanç, gider, sentroitler ve anahtar değerini parametre olarak alıyor
        
        #kayıt yapma seçimi için radio button
        if self.text_save.isChecked(): #1.radio buton seçili ise demek
            path_name = "kümeleme_sonucu.txt" #kayıt yapılacak dosya adı belirlendi
            text_file = open(path_name,"w") #yazma için dosya açar
            text_file.write(self.save_txt) #dosyaya sonucu yazar
            text_file.close() #dosyayı kapatır. Zorunludur.
            
        if self.plot_save.isChecked(): #2.radio buton seçili ise demek
            self.p.fig.savefig("kümeleme_sekil.jpg") #arayüzde bulunan şekli kayıt eder.
            
        if self.text_plot_save.isChecked():#3.radio buton seçili ise demek
        #yukarıdaki iki if'i de kapsar. sonucu .txt olarak yazar ve şekli kaydeder
            path_name = "kümeleme_sonucu.txt"
            text_file = open(path_name,"w")
            text_file.write(self.save_txt)
            text_file.close()  
            
            self.p.fig.savefig("kümeleme_sekil.jpg")
 
    def layouts(self):      
        # layout'ları oluşturma
        self.mainlayout = QHBoxLayout()
        
        self.leftlayout = QFormLayout()
        self.middlelayout = QFormLayout()
        self.rightlayout = QFormLayout()
        
        # soldaki layout'un tanımı
        self.leftlayoutGroupBox = QGroupBox("Plot") #group box tanımı, adı
        self.leftlayout.addRow(self.p)
        self.leftlayoutGroupBox.setLayout(self.leftlayout) #left layout'u group box'a ekleme işlemi
        
        # ortadaki layout'un tanımı
        self.middlelayoutGroupBox = QGroupBox("Kümeleme") #group box tanımı, adı
        self.middlelayout.addRow(self.k_number_text)
        self.middlelayout.addRow(self.k_number)
        self.middlelayout.addRow(self.text_save)
        self.middlelayout.addRow(self.plot_save)
        self.middlelayout.addRow(self.text_plot_save)
        self.middlelayout.addRow(self.cluster)
        self.middlelayoutGroupBox.setLayout(self.middlelayout) #middle layout'u group box'a ekleme işlemi
        
        # sağdaki layout'un tanımı
        self.rightlayoutGroupBox = QGroupBox("Sonuç") #group box tanımı, adı
        self.rightlayout.addRow(self.result_list)
        self.rightlayoutGroupBox.setLayout(self.rightlayout) #right layout'u group box'a ekleme işlemi
        
        # tüm layoutları main layaot'a tanımlayıp  tab'a ekleme işlemi
        self.mainlayout.addWidget(self.leftlayoutGroupBox,50) #yan yana sıralanırken nasıl ve ne oranda olacağını belirler
        self.mainlayout.addWidget(self.middlelayoutGroupBox,25)
        self.mainlayout.addWidget(self.rightlayoutGroupBox,25)
        
        self.tab1.setLayout(self.mainlayout) #main layout'u tab'a ekleme işlemi
        
        
class PlotCanvas(FigureCanvas):  #Plot bölümünün alt yapısının hazırlandığı sınıf
    
    def __init__(self, parent = None, width = 5, height = 5, dpi = 100): #dpi:inç başına düşen nokta sayısı
        
        self.fig = Figure(figsize=(width,height), dpi = dpi)
        
        FigureCanvas.__init__(self,self.fig)
        
        
    def plot(self, x,y,c,s, m = "."): #plot ettirmeye yarayan metot. (x ekseni,y ekseni, renk, boyut, plot üzerinde şeklin ifadesi)
        
        self.ax = self.figure.add_subplot(111) #(111:1.satır 1.sütuna 1şekil demek.) Subplot:  Grafiklerin düzlemini ve kaçıncı grafik olduğunu belirtir
        self.ax.scatter(x,y,c = c, s = s, marker = m) #scatter plot: iki farklı değer arasındaki ilişkiyi belirlemek için kullanılan ve noktalardan oluşan bir tablodur.
        self.ax.set_title("K-Means(Ortalama) Kümeleme")
        self.ax.set_xlabel("Gelir") #income
        self.ax.set_ylabel("İşlem Sayısı") #transaction
        self.draw() #plot'u çizdirmeye yarayan fonksiyon
        
    def clear(self): #şekli temizlemeye yarayan metot
        self.fig.clf()


             
pencere = Window() #pencere objesi ile window'u oluşturur.



