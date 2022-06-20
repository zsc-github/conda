import numpy as np
import math

class Dtmos(object):

    def __init__(self, granularity=60, window=0, sensitivity=0.95, max_trend_prop=0.0, upper_width=1,
                 lower_width=1, upper_detection=True, lower_detection=True, drop_suspicious=True,
                 min_value=-1.0, max_value=-1.0, compensate_mode='negative', compensate_coefficient=0.1,
                 seasonal='weekly',smooth_window=30,upper_constant=-1.0,lower_constant=-1.0):

        self.cursor = 0
        self.granularity = granularity / 60.0
        self.window = window
        self.sensitivity = sensitivity
        if max_trend_prop>1:
            self.max_trend_prop=1
        elif max_trend_prop<0:
            self.max_trend_prop=0
        else:
            self.max_trend_prop=max_trend_prop
        self.upper_width = upper_width
        self.lower_width = lower_width
        self.upper_detection = upper_detection
        self.lower_detection = lower_detection
        self.drop_suspicious = drop_suspicious
        self.min_value = min_value
        self.max_value = max_value
        self.compensate_mode = compensate_mode
        self.compensate_coefficient = max(0.0, compensate_coefficient)
        self.seasonal=seasonal
        self.smooth_window=smooth_window
        self.upper_constant=upper_constant
        self.lower_constant=lower_constant

        self.length_per_day = int(1440 / self.granularity)
        if self.window >= self.length_per_day:
            self.window = self.length_per_day - 1
        if self.window<0:
            self.window=0
        self.x_prime = 0.0
        self.x_magnitude = 0.0
        self.x_median = 0.0
        self.volatility_prime = 0.0  # 全局波动性均值
        self.volatility_var = 0.0  # 全局波动性方差
        self.volatility_likelihood = 0.0  # 波动性似然

        self.x = 0.0
        self.t = None
        self.estimation = 0.0  # 当前时间点拟合值
        self.estimation1=0.001
        self.historical_data = np.zeros([4, max(1,self.window * 2)])
        self.recent_data = np.zeros(max(1,self.window * 2))

        self.temp_estimation_length=min(60,max(self.window,1)*5)
        self.temp_magnitude_length=min(30,max(self.window,1)*5)
        self.temp_baseline_length=max(1,self.smooth_window)
        self.temp_data_length=self.length_per_day * 28 + max(1,self.window * 2)

        self.temp_estimation=np.zeros(self.temp_estimation_length)
        self.temp_magnitude=np.zeros(self.temp_magnitude_length)
        self.upper_array = np.zeros(self.temp_baseline_length)
        self.lower_array = np.zeros(self.temp_baseline_length)
        self.temp_data=np.zeros(self.temp_data_length)

        self.residual_prime = 0.0  # 全局残差均值
        self.residual_var = 0.0  # 全局残差方差
        self.error_likelihood = 0.0
        self.upper = 0.0
        self.lower = 0.0
        self.zscore = 0.0  # 异常得分
        self.anomaly = 0
        self.abnormality = 0
        self.model=dict()
        self.model_data=np.zeros(self.length_per_day * 29 + max(1,self.window * 2))

    def global_volatility(self, x):
        """
        :return: 更新全局波动性特征及波动性似然
        """
        self.x_prime = self.x_prime + (x - self.x_prime) / self.cursor if self.cursor != 0 else x
        volatility = np.abs(x - self.x_prime)
        volatility_u = self.volatility_prime
        self.volatility_prime += (volatility - self.volatility_prime) / self.cursor if self.cursor != 0 else 0
        diff = volatility - self.volatility_prime
        diff_u = volatility - volatility_u
        self.volatility_var += diff * diff_u
        volatility_sigma = np.sqrt(self.volatility_var / self.cursor) if self.cursor != 0 else 0
        z = diff / volatility_sigma if volatility_sigma != 0 else 0
        self.volatility_likelihood = 1 - 0.5 * math.erfc(abs(z) / 1.4142135623730951)

    def length_judgement(self):
        """
        :return: 选取历史信息
        """

        week_length = int(self.cursor / (self.length_per_day * 7))
        day_length = int(self.cursor / self.length_per_day)
        if self.seasonal=='daily':
            if day_length >= 4:
                for i in range(4):
                    self.historical_data[i, :] = self.temp_data[-(i + 1) * self.length_per_day - self.window:
                                                                -(i + 1) * self.length_per_day + max(self.window, 1)]
            else:
                for i in range(day_length):
                    self.historical_data[i, :] = self.temp_data[-(i + 1) * self.length_per_day - self.window:
                                                                -(i + 1) * self.length_per_day + max(self.window, 1)]
                if day_length == 0:
                    self.historical_data[0, :] = self.x_prime
                    self.historical_data[1, :] = self.x_median
                    self.historical_data[2, :] = self.x_magnitude
                if day_length == 1:
                    self.historical_data[1, :] = self.x_median
                    self.historical_data[2, :] = self.x_magnitude
                if day_length == 2:
                    self.historical_data[2, :] = self.x_magnitude
                self.historical_data[3, :] = self.temp_data[-max(1, 2 * self.window):]
        else:
            if week_length >= 4:
                for i in range(4):
                    self.historical_data[i, :] = self.temp_data[-(i + 1) * 7 * self.length_per_day - self.window:
                                                                -(i + 1) * 7 * self.length_per_day + max(self.window,1)]
            else:
                for i in range(week_length):
                    self.historical_data[i, :] = self.temp_data[-(i + 1) * 7 * self.length_per_day - self.window:
                                                                -(i + 1) * 7 * self.length_per_day + max(self.window,1)]
                if week_length == 0:
                    if day_length >= 4:
                        for i in range(4):
                            self.historical_data[i, :] = self.temp_data[-(i + 1) * self.length_per_day - self.window:
                                                                        -(i + 1) * self.length_per_day + max(self.window,1)]
                    else:
                        for i in range(day_length):
                            self.historical_data[i, :] = self.temp_data[-(i + 1) * self.length_per_day - self.window:
                                                                        -(i + 1) * self.length_per_day + max(self.window,1)]
                        if day_length == 0:
                            self.historical_data[0, :] = self.x_prime
                            self.historical_data[1, :] = self.x_median
                            self.historical_data[2, :] = self.x_magnitude
                        if day_length == 1:
                            self.historical_data[1, :] = self.x_median
                            self.historical_data[2, :] = self.x_magnitude
                        if day_length == 2:
                            self.historical_data[2, :] = self.x_magnitude
                        self.historical_data[3, :] = self.temp_data[-max(1,2 * self.window):]

                if week_length == 1:
                    self.historical_data[1, :] = self.temp_data[-self.length_per_day * 6 - self.window:
                                                                -self.length_per_day * 6 + max(self.window,1)]
                    self.historical_data[2, :] = self.temp_data[-self.length_per_day * 2 - self.window:
                                                                -self.length_per_day * 2 + max(self.window,1)]
                    self.historical_data[3, :] = self.temp_data[-self.length_per_day - self.window:
                                                                -self.length_per_day + max(self.window,1)]
                if week_length == 2:
                    self.historical_data[2, :] = self.temp_data[-self.length_per_day * 9 - self.window:
                                                                -self.length_per_day * 9 + max(self.window,1)]
                    self.historical_data[3, :] = self.temp_data[-self.length_per_day - self.window:
                                                                -self.length_per_day + max(self.window,1)]
                if week_length == 3:
                    self.historical_data[3, :] = self.temp_data[-self.length_per_day - self.window:
                                                                -self.length_per_day + max(self.window,1)]
        self.recent_data = self.temp_data[-max(1,2 * self.window):]

    def online_estimation(self):
        """
        :return: 实时计算拟合值,并计算异常程度
        """
        hist_median = np.median(np.median(self.historical_data, axis=0))
        recent_median = np.median(self.recent_data)

        self.estimation = (1 - self.max_trend_prop) * hist_median + self.max_trend_prop * recent_median
        self.temp_estimation=np.append(self.temp_estimation,self.estimation)[-self.temp_estimation_length:]
        if round(self.estimation,3)>0:
            self.estimation1=self.estimation
        residual = self.x - self.estimation
        residual_u = self.residual_prime
        self.residual_prime += (residual - self.residual_prime) / self.cursor if self.cursor != 0 else 0
        diff = residual - self.residual_prime
        diff_u = residual - residual_u
        self.residual_var += diff * diff_u
        self.residual_sigma = np.sqrt(self.residual_var / self.cursor) if self.cursor != 0 else 0
        self.zscore = diff / self.residual_sigma if self.residual_sigma != 0 else 0
        self.error_likelihood = 1 - 0.5 * math.erfc(abs(self.zscore)/ 1.4142135623730951)
        self.abnormality = abs(self.zscore) * 10
        if self.abnormality >= 100:
            self.abnormality = 99
        if self.abnormality > 0 and self.abnormality < 1:
            self.abnormality = 1

    def baseline_compensation(self):
        '''
        基线补偿
        :return:
        '''
        if self.min_value<0 or self.max_value<=0 or self.min_value>=self.max_value:
            compensate=0.0
            base=self.x_median
        else:
            if self.estimation>self.max_value:
                nor_fit=1
            elif self.estimation<self.min_value:
                nor_fit=0
            else:
                nor_fit=(self.estimation-self.min_value)/(self.max_value-self.min_value)
            if self.compensate_mode=='negative':
                compensate_degree=1-nor_fit
                compensate_base = max((self.max_value - self.estimation),0.0) * self.compensate_coefficient*compensate_degree
                compensate=compensate_degree*compensate_base
                base = self.x_median * compensate_degree
            elif self.compensate_mode=='positive':
                compensate_degree = nor_fit
                compensate_base = max((self.max_value - self.estimation),0.0) * self.compensate_coefficient*compensate_degree
                compensate=compensate_degree*compensate_base
                base = self.x_median * compensate_degree
            elif self.compensate_mode=='both':
                if nor_fit==0.5:
                    compensate_degree=1
                elif nor_fit>0.5:
                    compensate_degree=1-nor_fit
                else:
                    compensate_degree=nor_fit
                compensate_base = max((self.max_value - self.estimation),0.0) * self.compensate_coefficient*compensate_degree
                compensate=compensate_degree*compensate_base
                base = self.x_median * compensate_degree
            else:
                compensate=0.0
                base = self.x_median
        return base,compensate

    def get_baseline(self):
        '''
        :return: 计算上下基线
        '''
        self.x_magnitude = (np.mean(self.temp_estimation) + np.median(self.temp_estimation)) / 2
        if round(self.x_magnitude,4)==0:
            if round(self.estimation,4)!=0:
                self.x_magnitude=self.estimation
            else:
                self.x_magnitude=self.estimation1
        else:
            self.x_magnitude=round(self.x_magnitude,4)
        self.temp_magnitude=np.append(self.temp_magnitude,self.x_magnitude)[-self.temp_magnitude_length:]
        self.x_median=np.median(self.temp_magnitude)
        base,compensate=self.baseline_compensation()
        if self.cursor <= 2 * self.length_per_day:
            self.upper = 0.0
            self.lower = 0.0
        else:
            self.upper = self.estimation + self.upper_width * base+compensate
            self.lower = max(0, self.estimation - self.lower_width * base-compensate)
            if self.upper_constant!=-1.0 and self.upper>self.upper_constant:
                self.upper=self.upper_constant
            if self.lower_constant!=-1.0 and self.lower<self.lower_constant:
                self.lower=self.lower_constant
        self.upper_array = np.append(self.upper_array, self.upper)[-self.temp_baseline_length:]
        self.lower_array = np.append(self.lower_array, self.lower)[-self.temp_baseline_length:]
        self.upper = np.max(self.upper_array)
        self.lower = np.min(self.lower_array)
        self.upper = round(self.upper,4)
        self.lower = round(self.lower,4)

    def anomaly_detection(self):
        """
        :return: 异常检测
        """

        if self.cursor <=(self.length_per_day*2):
            self.anomaly=0
        else:
            if not self.drop_suspicious:
                if self.error_likelihood>=self.sensitivity:
                    if self.zscore>0:
                        self.upper=0.95*self.x
                    else:
                        self.lower=1.05*self.x

            if self.x>self.upper or self.x<self.lower:
                self.anomaly=1
            else:
                self.anomaly=0

            if self.x > self.upper and not self.upper_detection:
                self.anomaly = 0
            if self.x < self.lower and not self.lower_detection:
                self.anomaly = 0

        self.abnormality = int(0) if self.anomaly == 0 else self.abnormality
        if self.anomaly==1 and self.abnormality==0:
            self.abnormality=int(1)

    def fit(self, data):

        self.t = data['@timestamp']
        self.x = data['@value']
        self.global_volatility(self.x)
        self.length_judgement()
        self.online_estimation()
        self.get_baseline()
        self.anomaly_detection()
        self.temp_data=np.append(self.temp_data,self.x)[-self.temp_data_length:]
        self.cursor = self.cursor + 1
        result = dict(timestamp=self.t, value=self.x, pre_value=self.estimation, upper=self.upper,
                      lower=self.lower, anomaly=self.anomaly, abnormality=self.abnormality)
        return result

    @staticmethod
    def run(df,params):
        train_data=df.loc[:,[params["time_field"],params["value_field"]]]
        train_data.columns=['@timestamp','@value']
        dtmos_alg=Dtmos(granularity=params.get("granularity",60),
                        window=params.get("window",0),
                        sensitivity=params.get("sensitivity",0.95),
                        max_trend_prop=params.get("max_trand_prop",0.0),
                        upper_width=params.get("upper_width",0.5),
                        lower_width=params.get("lower_width",0.5),
                        upper_detection=params.get("upper_detection",True),
                        lower_detection=params.get("lower_detection",True),
                        drop_suspicious=params.get("drop_suspicious",True),
                        min_value=params.get("min_value",-1.0),
                        max_value=params.get("max_value",-1.0),
                        compensate_mode=params.get("compensate_mode","negative"),
                        compensate_coefficient=params.get("compensate_coefficient",0.1),
                        seasonal=params.get("seasonal",'weekly'),
                        smooth_window=params.get("smooth_window",30),
                        upper_constant=params.get("upper_constant",-1.0),
                        lower_constant=params.get("lower_constant",-1.0))
        pre_value = np.zeros(len(train_data), dtype=float)
        upper = np.zeros(len(train_data), dtype=float)
        lower = np.zeros(len(train_data), dtype=float)
        anomaly = np.zeros(len(train_data), dtype=int)
        abnormality = np.zeros(len(train_data), dtype=int)
        for idx, data in train_data.iterrows():
            detect_result = dtmos_alg.fit(data)
            pre_value[idx] = detect_result["pre_value"]
            upper[idx] = detect_result["upper"]
            lower[idx] = detect_result["lower"]
            anomaly[idx] = detect_result["anomaly"]
            abnormality[idx] = detect_result["abnormality"]
        df['pre_value'] = pre_value
        df['upper'] = upper
        df['lower'] = lower
        df['anomaly'] = anomaly
        df['abnormality'] = abnormality
        return df

if __name__ == '__main__':

    import pandas as pd
    df = pd.read_csv('/Users/enze/ALG_Zsc/jax-algorithm/kernel/test/seasonal_test_data.csv')
    time_field="@timestamp"
    value_field="@value"
    granularity=60
    dtmosAlg = Dtmos()
    params = {"time_field": time_field,
              "value_field": value_field,
              "granularity": granularity,
              "upper_width":0.25,
              "lower_width":0.5}
    result = Dtmos.run(df, params)
    import matplotlib.pyplot as plt

    result.index = pd.to_datetime(result[time_field])
    anomaly_index = result[result['anomaly'] == 1].index.tolist()
    plt.figure(figsize=(16, 8))
    plt.plot(result[value_field], color='k', label='value', alpha=0.75)
    plt.fill_between(result.index, result['upper'], result['lower'], color='#34A5DA', alpha=0.5, label='baseline')
    plt.scatter(anomaly_index, result.loc[anomaly_index, value_field], color='red', s=50, label='anomaly')
    plt.show()