#Univariate Time Series Prediction Model Framework:A Case Study Of Baoshan Temperature Forecasting in Shanghai)
#Highlights:Integrate GRU with single-head attention,genetic algorithm,first-order difference,and MLP as explicit trend branch
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from tqdm import tqdm
from datetime import datetime,timedelta
from sklearn.metrics import r2_score

NOAA_TOKEN="yours token"   #Key for web data scraping
plt.rcParams['font.sans-serif']=['SimHei','Microsoft YaHei','WenQuanYi Zen Hei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

STATION_ID="GHCND:CHM00058362"  #Weather station number
START_DATE="2023-01-01"
END_DATE="2026-01-15"

#Initialization of basic parameters of samples
SEQ_LEN=250   #Sample length
PRED_LEN=15   #Length of predicted series
TRAIN_RATIO=800  #Number of elements in the training set
#Parameters initialization related to genetic algorithm
POPULATION_SIZE=30
MAX_GENERATIONS=40  #The number of generations in genetic algorithm
TOURNAMENT_SIZE=5   #We employ the tournament selection strategy
ELITE_NUM=2    #Number of elites retained for the next generation
CROSSOVER_PROB=0.90    #Crossover probability
MUTATION_PROB=0.12     #Mutation probability
CV_FOLDS=5    #5-fold cross-validation

HYPERPARAM_SPACE={
    "hidden_dim":[[32,64,128],"discrete"],
    "num_layers":[[1,2,3],"discrete"],
    "batch_size":[[16,32,64],"discrete"],
    "epochs":[[20,30,50],"discrete"],
    "lr":[[1e-4,1e-3],"continuous"],
    "dropout_rate":[[0.1,0.3],"continuous"]
}

#Save the final result to a different file
OUTPUT_CSV="shanghai_baoshan_temperature_2023_2026.csv"
SEQ_NPZ="temperature_sequences.npz"
BEST_HP_NPZ="best_hyperparams.npz"
MODEL_PATH="final_temperature_model.pth"
EVAL_RESULT_NPZ="model_evaluation_results.npz"
LOSS_HISTORY_NPZ="training_loss_history.npz"
PREDICTION_CSV="2026_01_15_temperature_prediction.csv"
FIG_RAW_DATA="01_2023_2026_raw_temperature.png"
FIG_TRAIN_LOSS="03_training_validation_loss.png"
FIG_PRED_15DAYS="04_15days_prediction_comparison.png"

#Fixed seed
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Core model architecture
class LightweightSelfAttention(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.W_q=nn.Linear(hidden_dim,hidden_dim)
        self.W_k=nn.Linear(hidden_dim,hidden_dim)
        self.W_v=nn.Linear(hidden_dim,hidden_dim)
        self.W_o=nn.Linear(hidden_dim,hidden_dim)
        self.scale=torch.sqrt(torch.tensor(hidden_dim,dtype=torch.float32))
    def forward(self,x):
        batch_size,seq_len,hidden_dim=x.shape
        Q=self.W_q(x)
        K=self.W_k(x)
        V=self.W_v(x)
        attention_scores=torch.matmul(Q,K.transpose(-2,-1))/self.scale.to(device)
        attention_weights=torch.softmax(attention_scores,dim=-1)
        attention_output=torch.matmul(attention_weights,V)
        output=self.W_o(attention_output)
        return output,attention_weights

class GRUWithLightAttentionAndTrend(nn.Module):
    def __init__(self,input_dim=1,hidden_dim=64,num_layers=2,output_dim=15,dropout_rate=0.2):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.output_dim=output_dim
        self.gru=nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
            dropout=dropout_rate if num_layers>1 else 0
        )
        self.self_attention=LightweightSelfAttention(hidden_dim)
        self.dropout=nn.Dropout(dropout_rate)
        self.fc=nn.Linear(hidden_dim,output_dim)  #Main prediction branch

        #Explicit trend branch(Two-layer perceptron)
        self.trend_extractor=nn.Sequential(
            nn.Linear(2,8),  #Input:mean,standard deviation
            nn.ReLU(),
            nn.Linear(8,output_dim)
        )
    def forward(self,x):
        #The main branch of GRU+ATTENTION
        gru_out,_=self.gru(x)
        attention_out,attention_weights=self.self_attention(gru_out)
        last_step_feature=attention_out[:,-1,:]
        last_step_feature=self.dropout(last_step_feature)
        main_pred=self.fc(last_step_feature) #(batch,15)

        #Trend branch
        last_10=x[:,-10:,0] #We take the data of last 10days to predict the trend
        mean_last10=torch.mean(last_10,dim=1,keepdim=True)
        std_last10=torch.std(last_10,dim=1,keepdim=True)
        trend_features=torch.cat([mean_last10,std_last10],dim=1) #(batch,2)
        trend_pred=self.trend_extractor(trend_features) #(batch,15)

        prediction=main_pred+trend_pred
        return prediction,attention_weights

#Module 1:Data Acquisition and Cleaning
def fetch_and_clean_data():
    print("="*80)
    print("[Step 1/6]Start acquiring and cleaning temperature data")
    print("="*80)
    base_url="https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    headers={"token":NOAA_TOKEN}
    all_data=[]
    max_retries=3

    data_ranges=[
        ("2023-01-01","2023-12-31"),
        ("2024-01-01","2024-12-31"),
        ("2025-01-01","2025-12-31"),
        ("2026-01-01","2026-01-15")
    ]
    for batch_start,batch_end in date_ranges:
        print("Fetching time period:{batch_start} to {batch_end}")
        params={
            "datasetid":"GHCND",
            "stationid":STATIOIN_ID,
            "datatypeid":"TMAX",
            "startdate":batch_start,
            "enddate":batch_end,
            "limit":1000,
            "offset":0
        }
        while True:
            retry_count=0
            response=None
            while retry_count<max_retries:
                try:
                    response=requests.get(base_url,params=params,headers=headers,timeout=30)
                    if response.status_code==200:
                        break
                except Exception as e:
                    print(f"Request failed,retrying {retry_count+1}/{max_retries},Error:{e}")
                retry_count+=1
                time.sleep(1)  #Avoid detection of abnormal refresh
            if response is None or response.status_code!=200:
                raise Exception(f"NOAA API request failed,Error code:{response.status_code if response else 'No Response'}")

            data_json=response.json()
            if "results" not in data_json or len(data_json["results"])==0:
                break
            all_data.extend(data_json["results"])
            print(f"Batch Progress:Accumulated {len(all_data)} raw data records acquired...")
            params["offset"]+=params["limit"]
            time.sleep(0.2)

    #Data cleaning
    df_raw=pd.DataFrame(all_data)
    df_clean=df_raw[["date","value"]].copy()
    df_clean.columns=["Date","Daily Maximum Temperature(0.1℃)"]
    df_clean["Daily Maximum Temperature(℃)"]=df_clean["Daily Maximum Temperature(0.1℃)"]/10.0
    df_clean=df_clean[["Date","Daily Maximum Temperature(℃)"]]
    df_clean["Date"]=pd.to_datetime(df_clean["Date"]).dt.date
    full_date_range=pd.date_range(start=START_DATE,end=END_DATE).date
    df_full_dates=pd.DataFrame({"Date":full_date_range})
    df_clean=pd.merge(df_full_dates,df_clean,on="Date",how="left")
    temp_series=df_clean["Daily Maximum Temperature(℃)"]
    min_valid_temp=-15.0
    max_valid_temp=42.0

    bad_data_mask=(
        temp_series.isna()
        | (temp_series<min_valid_temp)
        | (temp_series>max_valid_temp)
    )

    temp_for_rolling=temp_series.mask(bad_data_mask)
    rolling_filled=temp_for_rolling.fillna(
        temp_for_rolling.rolling(window=7,min_periods=1,center=True).mean()
    )
    final_filled=rolling_filled.ffill().bfill()
    df_clean["Daily Maximum Temperature(℃)"]=final_filled

    total_bad=bad_data_mask.sum()
    if total_bad>0:
        print(f"Found {total_bad} bad data points(missing/anomalous value),which hanve been filled")
    df_clean=df_clean.sort_values("Date").reset_index(drop=True)
    df_clean.to_csv(OUTPUT_CSV,index=False,encoding="utf-8-sig")

    plt.figure(figsize=(16,6),dpi=160)
    plt.plot(pd.to_datetime(df_clean["Date"]),df_clean["Daily Maximum Temperature(℃)"],linewidth=0.8)
    plt.title("2023-2026,Maximum Temperature in Baoshan District,Shanghai,including Predictions")
    plt.grid(alpha=0.3)
    plt.savefig(FIG_RAW_DATA,bbox_inches='tight')
    plt.close()

    print(f"\nFinished acquiring and cleaning!{len(df_clean)} days of complete data in total")
    print(f"Data have been saved as:{OUTPUT_CSV}")
    print("="*80+"\n")
    return df_clean

#Module 2:Sliding window sample construction(Version of Difference)
def build_sequences(df):
    print("="*80)
    print("[Step 2/6]Start constructing sliding window sample(first-order difference)")
    print("="*80)

    df["Date"]=pd.to_datetime(df["Date"])
    df_history=df[(df["Date"]>="2023-01-01")&(df["Date"]<="2025-12-31")].copy()
    df_final_verify=df[(df["Date"]>="2026-01-01")&(df["Date"]<="2026-01-15")].copy()
    history_temp=df_histore["Daily Maximum Temperature(℃)"].values.astype(np.float32)
    final_verify_temp=df_final_verify["Daily Maximum Temperature(℃)"].values.astype(np.float32)

    #First-order difference
    diff_history=np.diff(history_temp)  #length=len(history_temp)-1
    train_diff=diff_history[:TRAIN_RATIO]
    test_diff=diff_history[TRAIN_RATIO:]
    #Stardardization
    train_mean=np.mean(train_diff)
    train_std=np.std(train_diff)
    train_norm=(train_diff-train_mean)/train_std
    test_norm=(test_diff-train_mean)/train_std

    def create_sequences(data):
        X,y=[],[]
        max_start_idx=len(data)-SEQ_LEN-PRED_LEN
        for start_idx in range(max_start_idx+1):
            input_seq=data[start_idx:start_idx+SEQ_LEN]
            target_seq=data[start_idx+SEQ+LEN:start_idx+SEQ_LEN+PRED_LEN]
            X.append(input_seq)
            y.append(target_seq)
        return np.array(X,dtype=np.float32),np.array(y,dtype=np.float32)
    X_train,y_train=create_sequences(train_norm)
    X_test,y_test=create_sequences(test_norm)

    np.savez(
        SEQ_NPZ,
        X_train=X_train,y_train=y_train,
        X_test=X_test,y_test=y_test,
        train_mean=train_mean,train_std=train_std,
        history_temp=history_temp,
        diff_history=diff_history,
        final_verify_temp=final_verify_temp
    )
    print(f"[DEBUG]Whether X_train has NaN:{np.isnan(X_train).any()},whether has Inf:{np.isinf(X_train).any()}")
    print(f"[DEBUG]Whether y_train has NaN:{np.isnan(y_train).any()},whether has Inf:{np.isinf(y_train).any()}")
    print(f"Sample building finished!")
    print(f"Training set sample count:{len(X_train)},Input shape:{X_train.shape},Label shape:{y_train.shape}")
    print(f"Test set sample count:{len(X_test)},Input shape:{X_test.shape},Label shape:{y_test.shape}")
    print(f"Sample has been saved as:{SEQ_NPZ}")
    print("="*80+"\n")
    return (X_train,y_train,X_test,y_test,train_mean,train_std,history_temp
            diff_history,final_verify_temp)

#Module 3:Genetic Algorithm Hyperparameter Optimization
def genetic_algorithm_optimization(X_train,y_train):
    print("="*80)
    print("[Step 3/6]Start GA Hyperparameter Optimization")
    print(f"Population size:{POPULATION_SIZE},The number of generations:{MAX_GENERATIONS}")
    print("="*80)

    def rolling_forward_cv(X,y,hyperparams):
        hidden_dim=hyperparams["hidden_dim"]
        num_layers=hyperparams["num_layers"]
        batch_size=hyperparams["batch_size"]
        epochs=hyperparams["epochs"]
        lr=hyperparams["lr"]
        dropout_rate=hyperparams["dropout_rate"]

        fold_length=len(X)//CV_FOLDS
        loss_list=[]

        for fold in range(CV_FOLDS-1):
            train_end_idx=(fold+1)*fold_length
            X_train_fold=X[:train_end_idx]
            y_train_fold=y[:train_end_idx]
            val_start_idx=train_end_idx
            val_end_idx=(fold+2)*fold_length
            X_val_fold=X[val_start_idx:val_end_idx]
            y_val_fold=y[val_start_idx:val_end_idx]

            X_train_tensor=torch.tensor(X_train_fold,dtype=torch.float32).unsqueeze(-1).to(device)
            y_train_tensor=torch.tensor(y_train_fold,dtype=torch.float32).to(device)
            X_val_tensor=torch.tensor(X_val_fold,dtype=torch.float32).unsqueeze(-1).to(device)
            y_val_tensor=torch.tensor(y_val_fold,dtype=torch.float32).to(device)

            train_dataset=torch.utils.data.TensorDataset(X_train_tensor,y_train_tensor)
            train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

            model=GRUWithLightAttentionAndTrend(
                hidden_dim=hidden_dim,num_layers=num_layers,dropout_rate=dropout_rate
            ).to(device)
            optimizer=optim.Adam(model.parameters(),lr=lr)
            criterion=nn.HuberLoss()   #Allow volatility

            model.train()
            for epoch in range(epochs):
                for batch_X,batch_y in train_loader:
                    optimizer.zero_grad()
                    pred,_=model(batch_X)
                    loss=criterion(pred,batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                val_pred,_=model(X_val_tensor)
                val_loss=criterion(val_pred,y_val_tensor).item()
                loss_list.append(val_loss)
        return np.mean(loss_list)

    def init_population(pop_size):
        population=[]
        for _ in range(pop_size):
            individual={}
            for param_name,(param_range,param_type) in HYPERPARAM_SPACE.items():
                if param_type=="discrete":
                    individual[param_name]=random.choice(param_range)
                else:
                    individual[param_name]=random.uniform(param_range[0],param_range[1])
            population.append(individual)
        return population

    def calculate_fitness(individual,X,y):
        loss=rolling_forward_cv(X,y,individual)
        return 1/(loss+ 1e-6),loss

    def tournament_selection(population,fitness_scores):
        selected_idx=random.sample(range(len(population)),TOURNAMENT_SIZE)
        selected_fitness=[fitness_scores[i] for i in selected_idx]
        best_idx=selected_idx[np.argmax(selected_fitness)]
        return population[best_idx]

    def crossover(parent1,parent2):
        child2,child2=parent1.copy(),parent2.copy()
        for param_name,(param_range,param_type) in HYPERPARAM_SPACE.items():
            if random.random()<0.55:
                if param_type=="discrete":
                    child1[param_name],child2[param_name]=parent2[param_name],parent1[param_name]
                else:
                    alpha=random.uniform(0.2,0.8)
                    child1[param_name]=np.clip(alpha*parent1[param_name]+(1-alpha)*parent2[param_name],param_range[0],param_range[1])
        return child1,child2

    def mutation(individual):
        mutated=individual.copy()
        for param_name,(param_range,param_type) in HYPERPARAM_SPACE.items():
            if random.random()<MUTATION_PROB:
                if param_type=="discrete":
                    current=mutated[param_name]
                    other_values=[v for v in param_range if v!=current]
                    mutated[param_name]=random.choice(other_values)
                else:
                    current=mutated[param_name]
                    sigma=(param_range[1]-param_range[0])*0.08
                    mutated_value=current+random.gauss(0,sigma)  #We apply Additive Gaussian Noise Mutation
                    mutated[param_name]=np.clip(mutated_value,param_range[0],param_range[1])
        return mutated

    population=init_population(POPULATION_SIZE)
    best_fitness_history=[]
    best_loss_history=[]
    best_individual_history=[]

    for generation in range(MAX_GENERATIONS):
        print(f"[Generation({generation+1}/{MAX_GENERATIONS})Calculating population fitness...")
        fitness_list=[]
        loss_list=[]
        for individual in tqdm(population):
            fitness,loss=calculate_fitness(individual,X_train,y_train)
            fitness_list.append(fitness)
            loss_list.append(loss)
        current_best_idx=np.argmax(fitness_list)
        current_best_fitness=fitness_list[current_best_idx]
        current_best_loss=loss_list[current_best_idx]
        current_best_individual=population[current_best_idx]

        best_fitness_history.append(current_best_fitness)
        best_loss_hostory.append(current_best_loss)
        best_individual_history.append(current_best_individual)

        print(f"Generation {generation+1} completed|Current optimal LOSS:{current_best_loss:.4f}")
        print(f"Current optimal hyperparameters:{current_best_individual}\n")

        next_generation=[]
        elite_indices=np.argsort(fitness_list)[-ELITE_NUM:]
        for idx in elite_indices:
            next_generation.append(population[idx])
        while len(next_generation)<POPULATION_SIZE:
            parent1=tournament_selection(population,fitness_list)
            parent2=tournament_selection(population,fitness_list)
            if random.random()<CROSSOVER_PROB:
                child1,child2=crossover(parent1,parent2)
            else:
                child1,child2=parent1.copy(),parent2.copy()
            child=mutation(child1)
            child=mutation(child2)
            next_generation.append(child1)
            if len(next_generation)<POPULATION_SIZE:
                next_generation.append(child2)
        population=next_generation

    global_bext_idx=np.argmax(best_fitness_history)
    global_best_loss=best_loss_history[global_best_idx]
    global_best_individual=best_individual_history[global_best_idx]

    np.savez(
        BEST_HP_NPZ,
        **global_best_individual,
        best_loss=global_best_loss,
        best_fitness_history=best_fitness_history,
        best_loss_history=best_loss_history
    )

    print("="*80)
    print("Genetic Algorithm Hyperparameter Optimization finished!")
    print(f"Global Best LOSS:{global_best_loss:.4f}")
    print(f"Global optimal hyperparameter combination:")
    for k,v in global_best_individual.items():
        if k in ["lr","dropout_rate"]:
            print(f"{k}:{v:.6f}")
        else:
            print(f"{k}:{v}")
    print(f"Optimal Hyperparameters have been saved as:{BEST_HP_NPZ}")
    print("="*80+"\n")
    return global_best_individual

#Module 4:Final Model Training and Test Set Evaluation
def train_and_evaluate_model(X_train,y_train,X_test,y_test,train_mean,train_std,best_hp):
    print("="*80)
    print("[Step 4/6]Start Final Model Training and Test Set Evaluation")
    print("="*80)

    batch_size=best_hp["batch_size"]
    epochs=best_hp["epochs"]
    lr=best_hp["lr"]

    X_train_tensor=torch.tensor(X_train,dtype=torch.float32).unsqueeze(-1).to(device)
    y_train_tensor=torch.tensor(y_train,dtype=torch.float32).to(device)
    X_test_tensor=torch.tensor(X_test,dtype=torch.float32).unsqueeze(-1).to(device)
    y_test_tensor=torch.tensor(y_test,dtype=torch.float32).to(device)
    train_dataset=torch.utils.data.TensorDataset(X_train_tensor,y_train_tensor)
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

    model=GRUWithLoghtAttentionAndTrend(
        hidden_dim=best_hp["hidden_dim"],
        num_layers=best_hp["num_layers"],
        dropout_rate=best_hp["dropout_rate"]
    ).to(device)

    print(f"Final model initialization completed,parameter quantity:{sum(p.numel() for p in model.parameters())}")
    optimizer=optim.Adam(model.parameters(),lr=lr)
    criterion=nn.HuberLoss()

    train_loss_history=[]
    print("Start Final Model Training...")
    for epoch in range(epochs):
        model.train()
        train_running_loss=0.0
        for batch_X,batch_y in tqdm(train_loader,desc=f"Epoch{epoch+1}/{epochs} training"):
            optimizer.zero_grad()
            pred,_=model(batch_X)
            loss=criterion(pred,batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            train_running_loss+=loss.item()*batch_X.size(0)
        avg_train_loss=train_running_loss/len(train_loader.dataset)
        train_loss_history.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{epochs} completed|Training HuberLoss:{avg_train_loss:.4f}\n")
    torch.save(model.state_dict(),MODEL_PATH)

    print("\nStart Test Set Evaluation...")
    model.eval()
    with torch.no_grad():
        test_pred,_=model(X_test_tensor)

    #Obtain differential predicted values via denormalization
    test_pred_diff=test_pred.cpu().numpy()*train_std+train_mean
    y_test_diff=y_test_tensor.cpu().numpy()*train_std+train_mean

    mse=np.mean((test_pred_diff-y_test_diff.flatten())
    rmse=np.sqrt(mse)
    r2=r2_score(y_test_diff.flatten(),test_pred_diff.flatten())

    np.savez(
        EVAL_RESULT_NPZ,
        test_pred_diff=test_pred_diff,
        y_test_diff=y_test_diff,
        mse=mse,
        rmse=rmse,
        r2=r2
    )
    np.savez(LOSS_HISTORY_NPZ,train_loss=train_loss_history)

    plt.figure(figsize=(12,5),dpi=160)
    plt.plot(train_loss_history,label="Training Loss Curve")
    plt.legend()
    plt.title("Training Convergence Curve(HuberLoss)")
    plt.savefig(FIG_TRAIN_LOSS,bbox_inches='tight')
    plt.close()

    print("="*80)
    print("Final Model Training and Evaluation completed!")
    print(f"Test set evaluation results(differential version):")
    print(f"1.MSE:{mse:.4f}")
    print(f"2.RMSE:{rmse:.4f}")
    print(f"3.R²:{r2:.4f}")
    print(f"Final Model has been saved as:{MODEL_PATH}")
    print("="*80+"\n")
    return model,mse,rmse,r2

#Module 5:Final prediction and verification for the next 15days(differential restoration)
def final_prediction(model,history_temp,diff_history,final_verify_temp,train_mean,train_std):
    print("="*80)
    print("[Step 5/6]Start Final forecast & verification(Jan.1-Jan.15,2026)|differential restoration")
    print("="*80)

    #Input the last SEQ_LEN difference values
    input_diff_raw=diff_history[-SEQ_LEN:]
    input_diff_norm=(input_diff_raw-train_mean)/train_std
    input_tensor=torch.tensor(input_diff_norm,dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

    model.eval()
    with torch.no_grad():
        pred_diff_norm,_=model(input_tensor)
    pred_diff=pred_diff_norm.cpu().numpy().flatten()*train_std+train_mean

    #Restore Original Temperature from Difference Values
    last_known_temp=history_temp[-1]
    pred_temps=[last_known_temp+pred_diff[0]]
    for i in range(1,PRED_LEN):
        pred_temps.append(pred_temps[-1]+pred_diff[i])
    pred_real=np.array(pred_temps)

    pred_datas=pd.date_range(start="2026-01-01",end="2026-01-15").date
    result_df=pd.DataFrame({
        "Date":pred_dates,
        "Predicted temperature(℃)":np.round(pred_real,2),
        "Real temperature(℃)":np.round(final_verify_temp,2),
        "Absolute error(℃)":np.round(np.abs(pred_real-final_verify_temp),2)
    })
    final_mse=np.mean((pred_real-final_verify_temp)**2)
    final_rmse=np.sqrt(final_mse)

    result_df.to_csv(PREDICTION_CSV,index=False,encoding="utf-8-sig")
    print("[Daily comparison of prediction results from 2026-01-01 to 2026-01-15]")
    print(result_df.to_string(col_space=20,index=False))
    print("="*80)

    plt.figure(figsize=(12,5),dpi=160)
    plt.plot(range(15),final_verify_temp,'g-o',label="True Curve")
    plt.plot(range(15),pred_real,'r--s',label="Predicted Curve")
    plt.title("15-Day Prediction Comparison(Differenced Model)")
    plt.legend()
    plt.savefig(FIG_PRED_15DAYS,bbox_inches='tight')
    plt.close()

    print(f"15-Day Overall accuracy of final predicton")
    print(f"MSE:{final_mse:.2f}℃")
    print(f"RMSE:{final_rmse:.2f}℃")
    print(f"The result of prediction has been saved as:{PREDICTION_CSV}")
    print("="*80+"\n")
    return result_df,final_mse,final_rmse

#Final Module:Main program
if __name__=="__main__":
    print("#"*90)
    print("One-click start of full process")
    print(f"Operating:{device}")
    print("#"*90+"\n")

    try:
        #Step 1:Data Acquisition and Cleaning
        df_data=fetch_and_clean_data()
        #Step 2:Construct ifferential Sample
        (X_train,y_train,X_test,y_test,
         train_mean,train_std,history_temp,diff_history,final_verify_temp)=build_sequences(df_data)
        #Step 3:Genetic Algorithm Hyperparameter Optimization
        best_hyperparams=genetic_algorithm_optimization(X_train,y_train)
        #Step 4:Final Model Training and Evaluation
        trained_model,test_mse,test_rmse,test_r2=train_and_evaluate_model(
            X_train,y_train,X_test,y_test,train_mean,train_std,best_hyperparams
        )
        #Step 5:Final prediction and verification
        final_result,final_mse,final_rmse=final_prediction(
            trained_model,history_temp,diff_history,final_verify_temp,train_mean,train_std
        )

        print("#"*90)
        print("Congratulations!All steps executed successfully")
        print("#"*90)
        print(f"[Final Result Summary]")
        print(f"1.MSE(differential version):{test_mse:.4f}")
        print(f"2.Final prediction MSE:{final_mse:.2f}℃")
        print(f"3.R²:{test_r2:.4f}")
        print("\n[List of all generated files]")
        print(f"Original temperature data:{OUTPUT_CSV}")
        print(f"Training sample data:{SEQ_NPZ}")
        print(f"Best hyperparameters:{BEST_HP_NPZ}")
        print(f"Trained final model:{MODEL_PATH}")
        print(f"Model Evaluation result:{EVAL_RESULT_NPZ}")
        print(f"Training loss history:{LOSS_HISTORY_NPZ}")
        print(f"15-DAY prediction result:{PREDICTION_CSV}")
        print("#"*90)
    except Exception as e:
        print("Full process execution error!!!")
        print(f"Error message:{e}")
        print("Check whether the Token is correct,the network is normal,and all dependent libraries are fully installed")
