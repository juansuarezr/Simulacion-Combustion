import numpy as np



R = 8.314 #J/mol.k

"""
ESPECIES

1. H20
2. O2
3. N2
4. CO
5. CO2
6. H2
7. CH4
8. H2S
9. NO
10. SO2

"""

matrix_coef_a1 = np.array([[4.198640560E+00, -2.036434100E-03, 6.520402110E-06, -5.487970620E-09, 1.771978170E-12, -3.029372670E+04, -8.490322080E-01], #H2O
                        [3.782456360E+00, -2.996734150E-03, 9.847302000E-06, -9.681295080E-09, 3.243728360E-12, -1.063943560E+03,   3.657675730E+00], #O2
                        [3.531005280E+00, -1.236609870E-04, -5.029994370E-07, 2.435306120E-09, -1.408812350E-12, -1.046976280E+03, 2.967474680E+0], #N2
                        [3.579533470E+00, -6.103536800E-04, 1.016814330E-06, 9.070058840E-10, -9.044244990E-13, -1.434408600E+04, 3.508409280E+00], #CO
                        [2.356773520E+00, 8.984596770E-03, -7.123562690E-06, 2.459190220E-09, -1.436995480E-13, -4.837196970E+04, 9.901052220E+00], #CO2
                        [2.344331120E+00, 7.980520750E-03, -1.947815100E-05, 2.015720940E-08, -7.376117610E-12, -9.179351730E+02, 6.830102380E-01], #H2
                        [5.149876130E+00, -1.367097880E-02, 4.918005990E-05, -4.847430260E-08, 1.666939560E-11, -1.024664760E+04, -4.641303760E+00], #CH4
                        [3.932347600E+00, -5.026090500E-04, 4.592847300E-06, -3.180721400E-09, 6.649756100E-13, -3.650535900E+03, 2.315790500E+00], #H2S
                        [4.218598960E+00, -4.639881240E-03, 1.104430490E-05, -9.340555070E-09, 2.805548740E-12,9.845099640E+03, 2.280610010E+00], #NO
                        [3.266533800E+00, 5.323790200E-03, 6.843755200E-07, -5.281004700E-09, 2.559045400E-12, -3.690814800E+04, 9.664651080E+00], #SO2
                        [-3.108720720E-01, 4.403536860E-03, 1.903941180E-06, -6.385469660E-09, 2.989642480E-12, -1.086507940E+02, 1.113829530E+00], #C
                        [-7.274056840E+01, 4.812225340E-01, -1.078422330E-03, 1.032577280E-06, -3.588844900E-10, 8.291348560E+03, 3.152697430E+02] #S
                        ]) 


species_list = ['H2O', 'O2', 'N2', 'CO', 'CO2', 'H2', 'CH4', 'H2S', 'NO', 'SO2', 'C', 'S']



class Gibbs:

    def __init__(self, species_list, matrix_coef_a1, T):
        """
        coef_cp, coef_entalpia y coef_entropia son matrices
        """
        self.species_list = species_list
        self.matrix_coef_a1 = matrix_coef_a1
        self.T = T

    def calor_especifico(self):
        """
        Toma como parámetros la matriz de coeficientes y la temperatura
        """
        cp_vector = []
        for i, j in enumerate(self.species_list):
            A1 = self.matrix_coef_a1[i][0]
            A2 = self.matrix_coef_a1[i][1]
            A3 = self.matrix_coef_a1[i][2]
            A4 = self.matrix_coef_a1[i][3]
            A5 = self.matrix_coef_a1[i][4]

            cp_vector.append((A1 + A2*self.T + A3*self.T**2 + A4*self.T**3 + A5*self.T**4)*R)

        return np.array(cp_vector)

    def entalpia_formacion(self):
        """
        Toma como parámetros la matriz de coeficientes y la temperatura
        """
        entalpia_vector = []
        for i, j in enumerate(self.species_list):
            A1 = self.matrix_coef_a1[i][0]
            A2 = self.matrix_coef_a1[i][1]
            A3 = self.matrix_coef_a1[i][2]
            A4 = self.matrix_coef_a1[i][3]
            A5 = self.matrix_coef_a1[i][4]
            A6 = self.matrix_coef_a1[i][5]

            entalpia_vector.append((A1 + (A2/2)*self.T + (A3/3)*self.T**2 + (A4/4)*self.T**3 + (A5/5)*self.T**4 + A6/self.T)*R*self.T)

        return np.array(entalpia_vector)


    def entropia_formacion(self):
        """
        Toma como parámetros la matriz de coeficientes y la temperatura
        """
        entropia_vector = []
        for i, j in enumerate(self.species_list):
            A1 = self.matrix_coef_a1[i][0]
            A2 = self.matrix_coef_a1[i][1]
            A3 = self.matrix_coef_a1[i][2]
            A4 = self.matrix_coef_a1[i][3]
            A5 = self.matrix_coef_a1[i][4]
            A7 = self.matrix_coef_a1[i][6]

            entropia_vector.append((A1*np.log(self.T) + A2*self.T + (A3/2)*self.T**2 + (A4/3)*self.T**3 + (A5/4)*self.T**4 + A7))

        entropia_formacion = []
        #['H2O', 'O2', 'N2', 'CO', 'CO2', 'H2', 'CH4', 'H2S', 'NO', 'SO2', 'C', 'S']
        entropia_formacion.append(entropia_vector[0]-entropia_vector[5]-entropia_vector[1]/2) #H2O
        entropia_formacion.append(0) #O2
        entropia_formacion.append(0) #N2
        entropia_formacion.append(entropia_vector[3]-entropia_vector[10]-entropia_vector[1]/2) #CO
        entropia_formacion.append(entropia_vector[4]-entropia_vector[10]-entropia_vector[1]) #CO2
        entropia_formacion.append(0) #H2
        entropia_formacion.append(entropia_vector[6]-entropia_vector[10]-2*entropia_vector[5]) #CH4
        entropia_formacion.append(entropia_vector[7]-entropia_vector[5]-entropia_vector[11]) #H2S
        entropia_formacion.append(entropia_vector[8]-entropia_vector[2]/2-entropia_vector[1]/2) #NO
        entropia_formacion.append(entropia_vector[9]-entropia_vector[11]-entropia_vector[1]) #SO2
        entropia_formacion.append(0) #C
        entropia_formacion.append(0) #S

        return np.array(entropia_formacion)

    
    def energia_gibbs(self):
        return self.entalpia_formacion() - self.T*self.entropia_formacion()


gibbs = Gibbs(species_list, matrix_coef_a1, T=500)
print(gibbs.energia_gibbs())

    