'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
'''
import numpy as np
import os
import pandas as pd
import sys
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

def truncated_avg(x, tol=0.1):
    """ function that compute the truncated mean of an array x with a tolerance tol 
        in order to ignore distant points
    
    Args:
        - x: two dimensional array
        - tol: pourcentage of tolearance arround the avreage

    Returns:
        - truncated mean of array x
    """

    X = np.tile(x[..., np.newaxis], (1, 1, x.shape[1]))
    Xbis = np.tile(x[:, np.newaxis, :], (1, x.shape[1], 1))
    range_X = np.logical_and(X >= Xbis-tol, X <= Xbis+tol)
    nbr_points = np.sum(range_X, axis=1)
    moy = np.sum(X*range_X, axis=1)/nbr_points
    max_points_idx = (nbr_points == np.tile(nbr_points.max(axis=1)[:,np.newaxis], (1, x.shape[1])))
    return np.sum(moy*max_points_idx, axis=1)/max_points_idx.sum(axis=1)

def reduce_sensor(flight):
    """ reduce and merge redundant sensors """

    sensor_list = ['JYIRS','JXIRS','JZIRS','QIRS','PIRS','RIRS','VZIRS','TETAIRS','PHIIRS','HDGTIRS','VEWIRS','VNSIRS','WAIRS']
    for sens in sensor_list:
        x = np.zeros((flight.shape[0],1))
        for i in range(3):
            if(sens+str(i+1)) in flight.columns:
                x = np.concatenate((x,np.expand_dims(flight[sens+str(i+1)],1)),axis=1)
                flight.drop(columns=sens+str(i+1),inplace=True)
        x = np.delete(x,0,axis=1) 
        avg = truncated_avg(x)
        flight.insert(flight.shape[1],sens,avg)

    sensor_list = ['MADS','ALFADS','BETADS','ZBPILADS','SATADS','PSTADS','TASADS']
    for sens in sensor_list:
        x = np.zeros((flight.shape[0],1))
        for i in range(4):
            if(sens+str(i+1)) in flight.columns:
                x = np.concatenate((x,np.expand_dims(flight[sens+str(i+1)],1)),axis=1)
                flight.drop(columns=sens+str(i+1),inplace=True)
        x = np.delete(x,0,axis=1) 
        avg = truncated_avg(x)
        flight.insert(flight.shape[1],sens,avg)
    
    
    # ordre pilote et co-pilote palonnier: Drapeau (queue), lacet (yaw)
    flight['DDNGM1'] = (flight['DDNGM1']+flight['DDNDM1'])/2
    flight.drop(columns='DDNDM1',inplace=True)

    # ordre pilote et co-pilote mini-mache: Gouvernes de profondeur (queue), tangage (pitch)
    flight['DDMGM1'] = flight['DDMGM1']+flight['DDMDM1']
    flight.drop(columns='DDMDM1',inplace=True)

    # ordre pilote et co-pilote mini-mache: Aileron (ailes), roulis (roll)
    flight['DDLGM1'] = flight['DDLGM1']+flight['DDLDM1']
    flight.drop(columns='DDLDM1',inplace=True)

    # Angles Spoiler (ailes) droit et gauche: angles symétrique, réduire la portance (descente)
    flight['DSPOILGM1'] = (flight['DSPOILGM1']+flight['DSPOILDM1'])/2
    flight.drop(columns='DSPOILDM1',inplace=True)

    return flight

def create_derivative_features_(df,num_vol=1):
    """ add derivative features of all sensors """
    X = np.zeros((df.shape[0],2*df.shape[1]),dtype=float)
    l = list(df.columns)
    for col in range(df.shape[1]):
        X[:,col] = df[l[col]].to_numpy()
        X[1:,df.shape[1]+col] = np.diff(X[:,col],axis=0)
        X[0,df.shape[1]+col] = X[1,df.shape[1]+col]
    return X

def create_features_history(df,hist_list):
    """ add history of all sensors on timestamps listed in hist_list """
    idx = np.array([hist_list],dtype=int)+np.arange(df.shape[0])[:,np.newaxis]
    idx = np.maximum(idx,0)
    X = np.zeros((df.shape[0],df.shape[1]*len(hist_list)+1),dtype=float)
    X[:,1:] = np.reshape(df[idx],(df.shape[0],-1))
    return X 

def data_preprocess(flight,flight_num,hist_list):
    """ data preprocessing : 
        (1) reduce redundant sensors
        (2) add derivative features 
        (3) add history
    """
    df1 = reduce_sensor(flight)
    df2 = create_derivative_features_(df1,flight_num)
    df2 = df2.astype(np.float32)
    df2 = create_features_history(df2,hist_list)
    df2[:,0] =  flight_num/200
    return df2



class model_Net(torch.nn.Module):

    def __init__(self,h):
        super(model_Net, self).__init__()
        self.ip_emb = torch.nn.Linear(158,256)
        self.relu1 = torch.nn.LeakyReLU(0.1) # torch.nn.PReLU()
        self.enc_lstm = torch.nn.LSTM(256,128,1)
        self.fc1 = torch.nn.Linear(129+256, 200) 
        self.relu2 = torch.nn.LeakyReLU(0.1) # torch.nn.PReLU()
        self.fc2 = torch.nn.Linear(200, 67)
        self.hist = h
        


    def forward(self, x):
        num_flight = torch.unsqueeze(x[:,0],1) # extract flight number feature

        # Features embedding
        x = x[:,1:].view((-1,self.hist,158))
        x = x.permute(1, 0, 2)
        embedding = self.relu1(self.ip_emb(x))

        # LSTM layer
        _,(hist_enc,_)  = self.enc_lstm(embedding)
        hist_enc = torch.squeeze(hist_enc,dim=0)
        
        # add flight number and last timestamp measurement to the history encoding
        x = torch.cat((num_flight,hist_enc,embedding[-1,:,:]),dim=1)

        # Fully connected layers
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return x


class model:
    def __init__(self):
        '''
        Constructor.
        '''
        self.input_dict = {'MADS1': 0, 'ALFADS1': 1, 'BETADS1': 2, 'ZBPILADS1': 3, 'SATADS1': 4, 'PSTADS1': 5, 'TASADS1': 6, 'MADS2': 7, 'ALFADS2': 8, 'BETADS2': 9, 'ZBPILADS2': 10, 'SATADS2': 11, 'PSTADS2': 12, 'TASADS2': 13, 'MADS3': 14, 'ALFADS3': 15, 'BETADS3': 16, 'ZBPILADS3': 17, 'SATADS3': 18, 'PSTADS3': 19, 'TASADS3': 20, 'MADS4': 21, 'ALFADS4': 22, 'BETADS4': 23, 'ZBPILADS4': 24, 'SATADS4': 25, 'PSTADS4': 26, 'TASADS4': 27, 'PCAB': 28, 'N1DSPLA': 29, 'N1MECLA': 30, 'N1DSPCA': 31, 'N1MECCA': 32, 'N1DSPRA': 33, 'N1MECRA': 34, 'FDPITCHCMD_FCS1_I': 35, 'FDROLLCMD_FCS1_I': 36, 'GWFMS1_I': 37, 'LAT_FMS1_I': 38, 'LON_FMS1_I': 39, 'JRFE11': 40, 'JRFM11': 41, 'JRFI11': 42, 'JRFE31': 43, 'JRFM31': 44, 'JRFI31': 45, 'JTOT1': 46, 'XTPGDN1': 47, 'XTAVDN1': 48, 'XTPDDN1': 49, 'XTPGUP1': 50, 'XTAVUP1': 51, 'XTPDUP1': 52, 'XABINRH_EXT_PANEL_GIO4A_I': 53, 'XABOUTLH_EXT_PANEL_GIO1A_I': 54, 'XABINLH_EXT_PANEL_GIO3A_I': 55, 'XABOUTRH_EXT_PANEL_GIO2A_I': 56, 'JYIRS1': 57, 'JXIRS1': 58, 'JZIRS1': 59, 'QIRS1': 60, 'PIRS1': 61, 'RIRS1': 62, 'VZIRS1': 63, 'TETAIRS1': 64, 'PHIIRS1': 65, 'HDGTIRS1': 66, 'VEWIRS1': 67, 'VNSIRS1': 68, 'WAIRS1': 69, 'WSFMS1_I': 70, 'JYIRS2': 71, 'JXIRS2': 72, 'JZIRS2': 73, 'QIRS2': 74, 'PIRS2': 75, 'RIRS2': 76, 'VZIRS2': 77, 'TETAIRS2': 78, 'PHIIRS2': 79, 'HDGTIRS2': 80, 'VEWIRS2': 81, 'VNSIRS2': 82, 'WAIRS2': 83, 'JYIRS3': 84, 'JXIRS3': 85, 'JZIRS3': 86, 'QIRS3': 87, 'PIRS3': 88, 'RIRS3': 89, 'VZIRS3': 90, 'TETAIRS3': 91, 'PHIIRS3': 92, 'HDGTIRS3': 93, 'VEWIRS3': 94, 'VNSIRS3': 95, 'WAIRS3': 96, 'XCTDN1': 97, 'XCTUP1': 98, 'XCTDN2': 99, 'XCTUP2': 100, 'DLGM1': 101, 'DLDM1': 102, 'DMGM1': 103, 'DMDM1': 104, 'DSPOILGM1': 105, 'DSPOILDM1': 106, 'DN1M1': 107, 'DDMGM1': 108, 'DDLGM1': 109, 'DDMDM1': 110, 'DDLDM1': 111, 'PHM1': 112, 'DVOLIG': 113, 'DVOLID': 114, 'DVOLEG': 115, 'DVOLED': 116, 'DAFLEVPOSM1': 117, 'TRIMMORDM1': 118, 'XSLATOUTLH': 119, 'XSLATOUTRH': 120, 'XSLATMIDLH': 121, 'XSLATMIDRH': 122, 'XSLATINTLH': 123, 'XSLATINTRH': 124, 'XAP1': 125, 'DDNGM1': 126, 'DDNDM1': 127, 'XWOWGRND_MWS1_I': 128, 'RA_RAD1A_I': 129}
        # self.output_dict_list = ['FFUS1', 'FFUS2', 'FFUS3', 'FFUS4', 'FFUS5', 'FFUS6', 'FFUS9', 'FFUS10', 'FFUS11', 'FFUS12', 'FFUS13', 'FFUS14', 'FFUS15', 'FFUS16', 'FFUS18', 'FFUS19', 'FWL2', 'FWL7', 'FWL9', 'FWL10', 'FWL11', 'FWL15', 'FWL17', 'FWL18', 'FWL19', 'FWL20', 'FWL25', 'FWL25BIS', 'FWR2', 'FWR7', 'FPHL1', 'FPHL2', 'FPHL3', 'FPHL4', 'FPHL5', 'FPHL6', 'FPHL7', 'FPHL9', 'FPHL10', 'FPHL11', 'FPHL12', 'FPHL13', 'FPHR1', 'FPHR2', 'FPHR3', 'FPHR4', 'FPHR5', 'FPHR6', 'FPHR7', 'FFIN2', 'FFIN3', 'FFIN10', 'FSCDLL1', 'FSCDLL2', 'FSCDLR1', 'FSCDLR2', 'FABAL2', 'FABAR2', 'FFLPAL1', 'FFLPAL2', 'FFLPAL3', 'FFLPAL4', 'FSLTAL1', 'FSLTAL3', 'FSLTAL4', 'FSCDN11', 'FSCDN12']
        self.output_dict  = {'FFUS1': 0, 'FFUS2': 1, 'FFUS3': 2, 'FFUS4': 3, 'FFUS5': 4, 'FFUS6': 5, 'FFUS9': 6, 'FFUS10': 7, 'FFUS11': 8, 'FFUS12': 9, 'FFUS13': 10, 'FFUS14': 11, 'FFUS15': 12, 'FFUS16': 13, 'FFUS18': 14, 'FFUS19': 15, 'FWL2': 16, 'FWL7': 17, 'FWL9': 18, 'FWL10': 19, 'FWL11': 20, 'FWL15': 21, 'FWL17': 22, 'FWL18': 23, 'FWL19': 24, 'FWL20': 25, 'FWL25': 26, 'FWL25BIS': 27, 'FWR2': 28, 'FWR7': 29, 'FPHL1': 30, 'FPHL2': 31, 'FPHL3': 32, 'FPHL4': 33, 'FPHL5': 34, 'FPHL6': 35, 'FPHL7': 36, 'FPHL9': 37, 'FPHL10': 38, 'FPHL11': 39, 'FPHL12': 40, 'FPHL13': 41, 'FPHR1': 42, 'FPHR2': 43, 'FPHR3': 44, 'FPHR4': 45, 'FPHR5': 46, 'FPHR6': 47, 'FPHR7': 48, 'FFIN2': 49, 'FFIN3': 50, 'FFIN10': 51, 'FSCDLL1': 52, 'FSCDLL2': 53, 'FSCDLR1': 54, 'FSCDLR2': 55, 'FABAL2': 56, 'FABAR2': 57, 'FFLPAL1': 58, 'FFLPAL2': 59, 'FFLPAL3': 60, 'FFLPAL4': 61, 'FSLTAL1': 62, 'FSLTAL3': 63, 'FSLTAL4': 64, 'FSCDN11': 65, 'FSCDN12': 66}
        self.missing_flight_list = ['205.VV','14.VV','156.VV','113.VV','143.VV','144.VV','153.VV','136.VV','155.VV','114.VV','11.VV','140.VV']
        self.is_trained = False
        
        self.ds = 6
        self.hist = 8
        self.hist_list = [-33,-27,-21,-15,-9,-6,-3,0] # list(range(-(self.hist-1)*self.ds,1,self.ds))
        self.torch_model = model_Net(len(self.hist_list))
        self.loss_func = F.mse_loss
        self.opt = torch.optim.Adam(self.torch_model.parameters(), lr=0.001)
        self.batch_size = 1024
        self.epoch_num = 1
        self.max_size = 600000

        

    def fit(self, flights):
        '''
        This function should train the model parameters on all flights.

        Args:
            flights: FlightDB object.
        '''

        for epoch in range(self.epoch_num):
            # shuffle flights order between epochs to add stochasticity  
            flight_shuffle = np.arange(len(flights))
            np.random.shuffle(flight_shuffle)

            # accumulate the samples of some number of flights untils reaching the maximum size of an array (600k of samples)
            i = 0
            while (i<1):#len(flights)):
                i0 = i
                s = [0]
                while(s[-1]<self.max_size)and((i<len(flights))):
                    flight_name = flights.keys[flight_shuffle[i]]
                    if flight_name not in self.missing_flight_list:
                        flight = flights[flight_name]
                        s.append(s[-1]+len(flight))
                    i = i+1                
                if(s[-1]>=self.max_size):
                    s.pop()
                    i = i-1
                i1 = i
                

                df_input0 = np.zeros((s[-1],79*2*self.hist+1),dtype=np.float32)
                df_output0 = np.zeros((s[-1],2*len(self.output_dict)),dtype=np.float32)
                j0 = 0
                for j in range(i0,i1):
                    flight_name = flights.keys[flight_shuffle[j]]
                    if flight_name not in self.missing_flight_list:
                        flight = flights[flight_name]
                        summary = flight.summary
                        
                        # determine missing input and output sensors
                        var_miss = summary['LABEL'][np.logical_not(summary['ENABLED'].astype(bool))] 
                        input_miss = [var_miss[i] for i in var_miss.index if summary['TYPE'][i]=="INPUT"] 
                        output_miss = [var_miss[i] for i in var_miss.index if summary['TYPE'][i]=="OUTPUT"]
                        flight_mask = np.ones((1,len(self.output_dict)),dtype=int) # create mask array for missing output sensors to ignore them in the cost function
                        
                        # fill missing input value with bachward fill method
                        df_input = flight.to_dataframe(flight.list_input)
                        df_input.interpolate(method='linear', axis=0, limit=None, inplace=True)
                        df_input.fillna(method='bfill', axis=0, inplace=True)
                        df_input.fillna(method='ffill', axis=0, inplace=True)
                        # replace missing input sensors with zeros
                        for var in input_miss:
                            df_input.insert(self.input_dict[var],var,np.zeros((df_input.shape[0])))
                        df_input = data_preprocess(df_input,int(flight.name[0:-3]),self.hist_list)
                        df_input = df_input.astype(np.float32)  

                        # fill missing output value with bachward fill method
                        df_output = flight.to_dataframe(flight.list_output)
                        df_output.interpolate(method='linear', axis=0, limit=None, inplace=True)
                        df_output.fillna(method='bfill', axis=0, inplace=True)
                        df_output.fillna(method='ffill', axis=0, inplace=True)
                        # replace missing output sensors with zeros
                        for var in output_miss:
                            df_output.insert(self.output_dict[var],var,np.zeros((df_output.shape[0])))
                            flight_mask[0,self.output_dict[var]] = 0 # update the mask array for missing output
                        df_output = df_output.to_numpy().astype(np.float32)
                        
                        # Concatenate the samples of each flight in a large array to add more stochasticity after shuffling
                        df_input0[s[j0]:s[j0+1],:] = df_input
                        df_output0[s[j0]:s[j0+1],:len(self.output_dict)] = df_output[:,:]
                        df_output0[s[j0]:s[j0+1],len(self.output_dict):] = flight_mask
                        j0 = j0+1
                
                # create Pytorch tensors and DataLoader that shuffle samples and divide data to batchs
                tensor_input = torch.tensor(df_input0)
                tensor_output = torch.tensor(df_output0)
                train_ds = TensorDataset(tensor_input, tensor_output)
                train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,num_workers=4)
                for xb, yb in train_dl:
                    pred = self.torch_model(xb)
                    pred = pred*yb[:,len(self.output_dict):] # apply mask on predected values
                    loss = self.loss_func(pred, yb[:,:len(self.output_dict)])
                    loss.backward()
                    self.opt.step()
                    self.opt.zero_grad()             
        self.is_trained = True        


    def predict(self, flight, outputs=None):
        '''
        Predict the values of the output variables y given the input variables X and the timestamps, for one specific flight.

        Args:
            flight: Flight object.
            outputs: Output variables used for evaluation (list of str).
        '''
        try:
            if outputs is None:
                outputs = flight.list_output
            flight_summary = flight.summary
            num_flight = int(flight.name[0:-3])

            # fill missing input value with linear interpolation and bachward fill method
            df_input = flight.to_dataframe(flight.list_input)
            df_input.interpolate(method='linear', axis=0, limit=None, inplace=True)
            df_input.fillna(method='bfill', axis=0, inplace=True)
            df_input.fillna(method='ffill', axis=0, inplace=True)

            # replace missing input sensors with zeros            
            var_miss = flight_summary['LABEL'][np.logical_not(flight_summary['ENABLED'].astype(bool))]
            input_miss = [var_miss[i] for i in var_miss.index if flight_summary['TYPE'][i]=="INPUT"]
            output_miss = [var_miss[i] for i in var_miss.index if flight_summary['TYPE'][i]=="OUTPUT"]
            for var in input_miss:
                df_input.insert(self.input_dict[var],var,np.zeros((df_input.shape[0])))
            df_input = data_preprocess(df_input,int(flight.name[0:-3]),self.hist_list)
            df_input = df_input.astype(np.float32)


            # predecting output sensor
            df_output_pred = np.zeros((df_input.shape[0],len(self.output_dict)))
            bch_size = self.batch_size*32
            with torch.no_grad():
                tensor_input = torch.tensor(df_input)
                test_ds = TensorDataset(tensor_input)
                test_dl = DataLoader(test_ds, batch_size=bch_size,num_workers=4)
                batch_num = 0
                for xb in test_dl:
                    xb = xb[0]
                    pred = self.torch_model(xb)
                    df_output_pred[batch_num*bch_size:batch_num*bch_size+pred.shape[0],:] = pred.detach().numpy()                
                    batch_num +=1

            # put predections in a dictionnary data structure as required
            predictions = {}
            for output in outputs: 
                predictions[output] = df_output_pred[:,self.output_dict[output]]

        except Exception as inst:
            print("Unexpected error: ",sys.exc_info())
            print("Flight Num: ",flight.name)
            traceback.print_exc(file=sys.stdout)
        return predictions