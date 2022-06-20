import numpy as np
import math
import copy
import time
from sklearn.neighbors.kde import KernelDensity



class Spider(object):

    def __init__(self,granularity=60,estimate_window=120,affected_window=0,likelihood_confidence=0.7,kde_weight=0.5,
                 bandwidth=1,kernel='gaussian',upper_width=1,lower_width=1,trend_prop=0.0,seasonal='',drop_suspicious=True,
                 upper_detection=True,lower_detection=True,min_value=-1,max_value=-1, compensate_mode='negative',
                 compensate_coefficient=0.1,smooth_window=30,fit_quantile=0.5,seasonal_anomaly_condition='day,7,3',
                 seasonal_anomaly_threshold=0.75,upper_constant=-1.0,lower_constant=-1.0):

        ## 数据计数器
        self.cursor=0

        ## 数据分钟颗粒度
        self.granularity=granularity/60.0

        ## 一天的数据量
        self.length_per_day=int(1440/self.granularity)

        ## 核密度估计长度，不小于10且不大于一天的数据量，[10,length_per_day]
        self.estimate_length=int(estimate_window/self.granularity)
        self.estimate_length=max(10,self.estimate_length)
        self.estimate_length=min(self.length_per_day,self.estimate_length)

        ## 拟合影响范围，不小于0且不大于一天的数据量，[0,length_per_day]
        self.affected_length=int(affected_window/self.granularity)
        self.affected_length=max(0,self.affected_length)
        self.affected_length=min(self.length_per_day,self.affected_length)

        ## 似然阈值
        self.likelihood_confidence=likelihood_confidence

        ## 核估计权重
        self.kde_weight=kde_weight

        ## 波动权重
        self.volatility_weight=(1-kde_weight)/2

        ## 核密度估计带宽
        self.bandwidth=bandwidth

        ## 核函数
        self.kernel=kernel

        ## 上基线宽度
        self.upper_width=upper_width

        ## 下基线宽度
        self.lower_width=lower_width

        ## 周期性参数
        self.seasonal=seasonal

        ## 趋势拟合权重，0～1之间，[0,1]
        if trend_prop>1:
            self.trend_prop=1
        elif trend_prop<0:
            self.trend_prop=0
        else:
            self.trend_prop=trend_prop

        ## 是否抛弃疑似异常点
        self.drop_suspicious=drop_suspicious

        ## 是否检测上基线
        self.upper_detection=upper_detection

        ## 是否检测下基线
        self.lower_detection=lower_detection

        ## 数据最大值
        self.min_value = min_value

        ## 数据最小值
        self.max_value = max_value

        ## 基带补偿模式
        self.compensate_mode = compensate_mode

        ## 基带补偿系数
        self.compensate_coefficient = max(0.0, compensate_coefficient)

        ## 基线平滑窗口
        self.smooth_window=smooth_window

        ## 拟合分位点
        self.fit_quantile=fit_quantile

        ## 周期性异常判定条件，形式为'day,7,3'的字符串
        self.seasonal_anomaly_condition=seasonal_anomaly_condition

        ## 周期性异常类型，day or week
        self.seasonal_type=self.seasonal_anomaly_condition.split(',')[0]

        ## 周期性异常检查周期的数量，不小于1
        self.check_season=int(self.seasonal_anomaly_condition.split(',')[1])
        self.check_season=max(1,self.check_season)

        ## 周期性异常出现周期的数量，不小于1
        self.occur_season=int(self.seasonal_anomaly_condition.split(',')[2])
        self.occur_season=max(1,self.occur_season)

        ## 周期性异常判定阈值
        self.seasonal_anomaly_threshold = seasonal_anomaly_threshold

        ## 上基线极大值
        self.upper_constant=upper_constant

        ## 下基线极小值
        self.lower_constant=lower_constant

        ## 观测时间，'%Y-%m-%d %H:%M:%S'格式的字符串
        self.t=''

        ## 观测值
        self.x=0.0

        ## 观测均值
        self.x_prime=0.0

        ## 观测中位数
        self.median=0.0

        ## 观测数量级
        self.x_magnitude=0.0

        ## 观测方差
        self.var=0.0

        ## 观测值似然
        self.value_likelihood=0.0

        ## 波动均值
        self.volatility_prime=0.0

        ## 波动方差
        self.volatility_var=0.0

        ## 波动似然
        self.volatility_likelihood=0.0

        ## 移动加权平均/非周期数据拟合值
        self.evma=0.0

        ## 移动加权平均系数
        self.beta = math.exp(-self.granularity /(estimate_window+self.affected_length*5+1))

        ## 周期数据拟合值
        self.estimation=0.0

        ## 残差
        self.residual=0.0

        ## 误差似然
        self.error_likelihood=0.0

        ## 预测值数组，长度固定为affected_length+60
        self.temp_pre_value_length = self.affected_length+60
        self.temp_pre_value = np.zeros(self.temp_pre_value_length)

        ## 观测值量纲数组，长度固定为affected_length+30
        self.temp_magnitude_length = self.affected_length+30
        self.temp_magnitude = np.zeros(self.temp_magnitude_length)

        ## 上下基线数组，长度为 max(1,smooth_window)
        self.temp_baseline_length = max(1,self.smooth_window)
        self.upper_array = np.zeros(self.temp_baseline_length)
        self.lower_array = np.zeros(self.temp_baseline_length)

        ## 历史数据数组，长度为 length_per_day*28+max(1,affected_length * 2)
        self.temp_data_length = self.length_per_day * 28 + max(1,self.affected_length * 2)
        self.temp_data = np.zeros(self.temp_data_length)

        ## evma误差数组，长度为 max(5,affected_length)
        self.temp_error = np.zeros(max(5, self.affected_length))

        ## 残差数组，长度固定为estimate_length
        self.temp_residual = np.zeros(self.estimate_length)

        ## 历史异常点日期，长度最大为28 * length_per_day
        self.anomaly_date=[]

        ## 历史信息数组，长度为max(1, affected_length * 2)]
        self.historical_data = np.zeros([4, max(1, self.affected_length * 2)])

        ## 趋势信息数组，长度为max(1, affected_length * 2)
        self.recent_data = np.zeros(max(1, self.affected_length * 2))

        ## 核估计概率
        self.kde_pro=0.0

        ## 拟合值
        self.pre_value=0.0

        ## 备用拟合值
        self.pre_value1=0.001

        ## 异常度
        self.abnormality=0.0

        ## 异常标签
        self.anomaly=0

        ## 上基线
        self.upper=0.0

        ## 下基线
        self.lower=0.0

        ## 当前点的模型
        self.model=dict()

        ## 上一个点模型
        self.last_model=dict()

        ## 上一个点模型历史数据数组
        self.last_model_data=np.zeros(self.temp_data_length)

        ## 模型历史数据数组
        self.model_data=np.zeros(self.length_per_day * 29 + max(1,self.affected_length * 2))

    def global_volatility(self,x):
        """
        :param x:
        :return: 更新全局波动性特征及波动性似然
        """
        volatility = np.abs(x - self.x_prime)
        volatility_u = self.volatility_prime
        self.volatility_prime += (volatility - self.volatility_prime) / self.cursor if self.cursor!=0 else 0
        diff = volatility - self.volatility_prime
        diff_u = volatility - volatility_u
        self.volatility_var += diff * diff_u
        volatility_sigma = np.sqrt(self.volatility_var / self.cursor) if self.cursor!=0 else 0
        z = diff / volatility_sigma if volatility_sigma != 0 else 0
        self.volatility_likelihood = 1 - 0.5 * math.erfc(abs(z) / 1.4142135623730951)

    def value_dist(self,x):
        '''
        :param x:
        :return: 更新观测值统计及似然
        '''
        u_prime=self.x_prime
        self.x_prime=self.x_prime+(x-self.x_prime)/self.cursor if self.cursor!=0 else x
        diff=x-self.x_prime
        diff_u=x-u_prime
        self.var=self.var+diff*diff_u
        sigma=np.sqrt(abs(self.var)/self.cursor) if self.cursor!=0 else 0
        z=(x-self.x_prime)/sigma if sigma!=0 else 0
        self.value_likelihood=1-0.5*math.erfc(abs(z)/1.4142135623730951)

    def error_dist(self,x):
        '''
        :param x:
        :return: 更新误差似然
        '''
        u_evma=self.evma
        self.evma=x+self.beta*(u_evma-x)
        self.temp_error=np.append(self.temp_error,x-self.evma)[-max(5,self.affected_length):]
        error_mean=self.temp_error.mean()
        error_std=self.temp_error.std()
        error_z=(x-self.evma-error_mean)/error_std if error_std!=0 else 0
        self.error_likelihood=1-0.5*math.erfc(abs(error_z)/1.4142135623730951)

    def length_judgement(self):
        """
        :return: 选取历史信息
        """
        ###优化1：支持天周期拟合
        ###优化2：选取历史信息时允许只选一个点
        ###优化3：根据拟合分位点计算拟合值

        week_length = int(self.cursor / (self.length_per_day * 7))
        day_length = int(self.cursor / self.length_per_day)

        if self.seasonal=='daily':
            if day_length >= 4:
                for i in range(4):
                    self.historical_data[i, :] = self.temp_data[-(i + 1) * self.length_per_day - self.affected_length:
                                                                -(i + 1) * self.length_per_day + max(self.affected_length, 1)]
            else:
                for i in range(day_length):
                    self.historical_data[i, :] = self.temp_data[-(i + 1) * self.length_per_day - self.affected_length:
                                                                -(i + 1) * self.length_per_day + max(self.affected_length, 1)]
                if day_length == 0:
                    self.historical_data[0, :] = self.x_prime
                    self.historical_data[1, :] = self.median
                    self.historical_data[2, :] = self.x_magnitude
                if day_length == 1:
                    self.historical_data[1, :] = self.median
                    self.historical_data[2, :] = self.x_magnitude
                if day_length == 2:
                    self.historical_data[2, :] = self.x_magnitude
                self.historical_data[3, :] = self.temp_data[-max(1, 2 * self.affected_length):]
        else:
            if week_length >= 4:
                for i in range(4):
                    self.historical_data[i, :] = self.temp_data[-(i + 1) * 7 * self.length_per_day - self.affected_length:
                                                                -(i + 1) * 7 * self.length_per_day + max(self.affected_length,1)]
            else:
                for i in range(week_length):
                    self.historical_data[i, :] = self.temp_data[-(i + 1) * 7 * self.length_per_day - self.affected_length:
                                                                -(i + 1) * 7 * self.length_per_day + max(self.affected_length,1)]
                if week_length == 0:
                    if day_length >= 4:
                        for i in range(4):
                            self.historical_data[i, :] = self.temp_data[-(i + 1) * self.length_per_day - self.affected_length:
                                                                        -(i + 1) * self.length_per_day + max(self.affected_length,1)]
                    else:
                        for i in range(day_length):
                            self.historical_data[i, :] = self.temp_data[-(i + 1) * self.length_per_day - self.affected_length:-(i + 1) * self.length_per_day + max(self.affected_length,1)]
                        if day_length == 0:
                            self.historical_data[0, :] = self.x_prime
                            self.historical_data[1, :] = self.median
                            self.historical_data[2, :] = self.x_magnitude
                        if day_length == 1:
                            self.historical_data[1, :] = self.median
                            self.historical_data[2, :] = self.x_magnitude
                        if day_length == 2:
                            self.historical_data[2, :] = self.x_magnitude
                        self.historical_data[3, :] = self.temp_data[-max(1,2 * self.affected_length):]

                if week_length == 1:
                    self.historical_data[1, :] = self.temp_data[-self.length_per_day * 6 - self.affected_length:
                                                                -self.length_per_day * 6 + max(self.affected_length,1)]
                    self.historical_data[2, :] = self.temp_data[-self.length_per_day * 2 - self.affected_length:
                                                                -self.length_per_day * 2 + max(self.affected_length,1)]
                    self.historical_data[3, :] = self.temp_data[-self.length_per_day - self.affected_length:
                                                                -self.length_per_day + max(self.affected_length,1)]
                if week_length == 2:
                    self.historical_data[2, :] = self.temp_data[-self.length_per_day * 9 - self.affected_length:
                                                                -self.length_per_day * 9 + max(self.affected_length,1)]
                    self.historical_data[3, :] = self.temp_data[-self.length_per_day - self.affected_length:
                                                                -self.length_per_day + max(self.affected_length,1)]
                if week_length == 3:
                    self.historical_data[3, :] = self.temp_data[-self.length_per_day - self.affected_length:
                                                                -self.length_per_day + max(self.affected_length,1)]
        self.recent_data = self.temp_data[-max(1,2 * self.affected_length):]
        row_meidan = np.percentile(self.historical_data, int(self.fit_quantile * 100), axis=0)
        hist_median = np.median(row_meidan)
        recent_median = np.median(self.recent_data)
        self.estimation = (1 - self.trend_prop) * hist_median + self.trend_prop * recent_median

    def baseline_compensation(self):
        '''
        基带补偿
        :return:
        '''
        if self.min_value < 0 or self.max_value <= 0 or self.min_value >= self.max_value:
            compensate = 0.0
            base = self.median
        else:
            if self.pre_value > self.max_value:
                nor_fit = 1
            elif self.pre_value < self.min_value:
                nor_fit = 0
            else:
                nor_fit = (self.pre_value - self.min_value) / (self.max_value - self.min_value)
            if self.compensate_mode == 'negative':
                compensate_degree = 1 - nor_fit
                compensate_base = max((self.max_value - self.pre_value),0.0) * self.compensate_coefficient * compensate_degree
                compensate = compensate_degree * compensate_base
                base = self.median * compensate_degree
            elif self.compensate_mode == 'positive':
                compensate_degree = nor_fit
                compensate_base = max((self.max_value - self.pre_value),0.0) * self.compensate_coefficient * compensate_degree
                compensate = compensate_degree * compensate_base
                base = self.median * compensate_degree
            elif self.compensate_mode == 'both':
                if nor_fit == 0.5:
                    compensate_degree = 1
                elif nor_fit > 0.5:
                    compensate_degree = 1 - nor_fit
                else:
                    compensate_degree = nor_fit
                compensate_base = max((self.max_value - self.pre_value),0.0) * self.compensate_coefficient * compensate_degree
                compensate = compensate_degree * compensate_base
                base = self.median * compensate_degree
            else:
                compensate = 0.0
                base = self.median
        return base, compensate

    def get_baseline(self):
        '''
        :return: 计算上下基线
        '''
        ###优化：基线补偿
        self.x_magnitude = (np.mean(self.temp_pre_value) + np.median(self.temp_pre_value)) / 2
        if round(self.x_magnitude, 4) == 0:
            if round(self.pre_value, 4) != 0:
                self.x_magnitude = self.pre_value
            else:
                self.x_magnitude = self.pre_value1
        else:
            self.x_magnitude = round(self.x_magnitude, 4)
        self.temp_magnitude = np.append(self.temp_magnitude, self.x_magnitude)[-self.temp_magnitude_length:]
        self.median = np.median(self.temp_magnitude)
        base,compensate = self.baseline_compensation()
        if self.cursor <= self.length_per_day:
            self.upper = 0.0
            self.lower = 0.0
        else:
            self.upper = self.pre_value + self.upper_width * base+compensate
            self.lower = max(0, self.pre_value - self.lower_width * base-compensate)
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

    def seasonal_anomaly_drop(self):
        '''
        非周期性数据周期异常消除
        :return:
        '''
        if self.min_value >= 0 and self.max_value > 0 and self.min_value < self.max_value:
            if self.x > self.max_value * self.seasonal_anomaly_threshold or self.x < self.min_value * (1 - self.seasonal_anomaly_threshold):
                drop_condition=False
            else:
                drop_condition=True
        else:
            drop_condition=True
        if self.anomaly == 1:
            if not drop_condition:
                pass
            else:
                current_seconds = time.mktime(time.strptime(self.t, '%Y-%m-%d %H:%M:%S'))
                delta_date = []
                delta_day = []
                for i in range(len(self.anomaly_date)):
                    date = self.anomaly_date[i]
                    if date != '':
                        seconds = time.mktime(time.strptime(self.anomaly_date[i], '%Y-%m-%d %H:%M:%S'))
                        if self.seasonal_type=='week':
                            for j in range(min(4,self.check_season)):
                                if abs(current_seconds-seconds)<=86400*(j+1)*7+self.smooth_window*60 and abs(
                                        current_seconds-seconds)>=86400*(j+1)*7-self.smooth_window*60:
                                    delta_date.append(date)
                                    delta_day.append(date[:10])
                        else:
                            for j in range(min(28,self.check_season)):
                                if abs(current_seconds-seconds)<=86400*(j+1)+self.smooth_window*60 and abs(
                                        current_seconds-seconds)>=86400*(j+1)-self.smooth_window*60:
                                    delta_date.append(date)
                                    delta_day.append(date[:10])
                delta_day = list(set(delta_day))
                if len(delta_day) >= self.occur_season:
                    self.anomaly = 0
                    if self.x - self.evma > 0:
                        self.upper = self.x * (1.05)
                        self.upper_array[-1]=self.upper
                        self.upper=np.max(self.upper_array)
                    else:
                        self.lower = self.x * (0.95)
                        self.lower_array[-1]=self.lower
                        self.lower=np.min(self.lower_array)
            self.anomaly_date = np.append(self.anomaly_date, self.t)[-self.length_per_day*28:]

    def anomaly_detdection(self):
        '''
        :return: 异常判定及异常度计算
        '''
        ## 优化：非周期性数据周期异常消除

        if self.value_likelihood<=self.likelihood_confidence and self.volatility_likelihood<=self.likelihood_confidence:
            self.abnormality=self.kde_weight*self.volatility_likelihood+(1-self.kde_weight)*self.value_likelihood
        else:
            self.abnormality=self.kde_weight*self.kde_pro+self.volatility_weight*(self.volatility_likelihood+self.value_likelihood)

        if self.cursor <=(self.length_per_day):
            self.anomaly=0
        else:
            if not self.drop_suspicious:
                if self.abnormality>=self.likelihood_confidence:
                    if self.residual>0:
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
        if self.seasonal_type!='no':
           self.seasonal_anomaly_drop()
        if self.upper_constant!=-1.0 and self.upper>self.upper_constant:
            self.upper=self.upper_constant
            if self.x>self.upper:
                self.anomaly=1
        if self.lower_constant!=-1.0 and self.lower<self.lower_constant:
            self.lower=self.lower_constant
            if self.x<self.lower:
                self.anomaly=1
        self.abnormality = int(self.abnormality * 100)
        if self.abnormality >= 100:
            self.abnormality = 99
        self.abnormality = int(0) if self.anomaly == 0 else self.abnormality
        if self.anomaly==1 and self.abnormality==0:
            self.abnormality=int(1)

    def array_reshape(self,array,length):
        '''
        数组重构,便于实时java版识别动态变量中的数组
        :param array:
        :param length:
        :return:
        '''
        temp_array=np.zeros(length)
        if self.cursor<=length:
            temp_array[:self.cursor]=array[-self.cursor:]
        else:
            remainder=self.cursor%length
            temp_array[:remainder]=array[length-remainder:]
            temp_array[remainder:]=array[:length-remainder]
        return temp_array

    def update_model(self):
        '''
        模型更新
        :return:
        '''
        if self.cursor==0:
            self.model=dict()
            self.model_data1=self.model_data
        else:
            self.model['cursor'] = self.cursor
            self.model['prime'] = self.x_prime
            self.model['magnitude'] = self.x_magnitude
            self.model['median'] = self.median
            self.model['var'] = self.var
            self.model['volatilityPrime'] = self.volatility_prime
            self.model['volatilityVar'] = self.volatility_var
            self.model['evma'] = self.evma
            self.model['preValue1'] = self.pre_value1
            # upper_array=self.array_reshape(self.upper_array,self.temp_baseline_length)
            # lower_array=self.array_reshape(self.lower_array,self.temp_baseline_length)
            # temp_residual=self.array_reshape(self.temp_residual,self.estimate_length)
            # temp_pre_value=self.array_reshape(self.temp_pre_value,self.temp_pre_value_length)
            # temp_magnitude=self.array_reshape(self.temp_magnitude,self.temp_magnitude_length)
            self.model_data1 = np.zeros(len(self.model_data))
            if self.cursor <= len(self.model_data):
                self.model_data1[:self.cursor] = self.model_data[-self.cursor:]
            else:
                self.model_data1 = self.model_data
            self.model['tempPreValue'] = self.temp_pre_value
            self.model['tempMagnitude'] = self.temp_magnitude
            self.model['tempResidual'] = self.temp_residual
            self.model['upperArray'] = self.upper_array
            self.model['lowerArray'] = self.lower_array
        self.last_model=copy.deepcopy(self.model)
        if self.cursor <= len(self.temp_data):
            self.last_model_data[:self.cursor] = self.temp_data[-self.cursor:]
        else:
            self.last_model_data= self.temp_data
        self.last_model["tempData"]=self.last_model_data

    def fit(self,data):

        self.t = data['@timestamp']
        self.x = data['@value']
        self.value_dist(self.x)
        self.global_volatility(self.x)
        self.error_dist(self.x)
        if self.seasonal=='daily' or self.seasonal=='weekly':
           self.length_judgement()
           self.pre_value = self.estimation
        else:
           self.pre_value=self.evma
        if round(self.pre_value,3)>0:
            self.pre_value1=self.pre_value
        self.residual=self.x-self.pre_value
        self.temp_pre_value=np.append(self.temp_pre_value,self.pre_value)[-self.temp_pre_value_length:]
        self.temp_residual=np.append(self.temp_residual,abs(self.residual))[-self.estimate_length:]
        if self.seasonal:
            estimate_data=self.temp_residual[-min(self.cursor,self.estimate_length):]
        else:
            estimate_data=self.temp_data[-min(self.cursor,self.estimate_length):]

        kde = KernelDensity(bandwidth=self.bandwidth,kernel=self.kernel)
        kde.fit(estimate_data.reshape(-1,1))
        score = kde.score_samples(estimate_data.reshape(-1,1))
        pos_score=abs(score)
        max_score=pos_score.max()
        min_score=pos_score.min()
        standard_score=[0+(0.99-0)/(max_score-min_score)*(s-min_score) if s!=min_score else 0 for s in pos_score]

        self.kde_pro=standard_score[-1]
        self.get_baseline()
        self.anomaly_detdection()
        self.temp_data = np.append(self.temp_data, self.x)[-self.temp_data_length:]
        self.model_data = np.append(self.model_data, self.x)[-self.length_per_day * 29 - self.affected_length * 2:]
        self.cursor = self.cursor + 1
        self.update_model()
        result = dict(timestamp=self.t, value=self.x, pre_value=self.pre_value, upper=self.upper,
                      lower=self.lower, anomaly=self.anomaly, abnormality=self.abnormality,
                      model=self.model,model_data=self.model_data1,last_model=self.last_model,
                      anomaly_date=list(self.anomaly_date))
        return result

    @staticmethod
    def run(df,params):
        train_data = df.loc[:, [params["time_field"], params["value_field"]]]
        train_data.columns = ['@timestamp', '@value']
        spider_alg = Spider(granularity=params.get("granularity",60),
                            estimate_window=params.get("estimate_window",60),
                            affected_window=params.get("affected_window",0),
                            likelihood_confidence=params.get("likelihood_confidence",0.85),
                            kde_weight=params.get("kde_weight",0.6),
                            bandwidth=params.get("bandwidth",1.0),
                            kernel=params.get("kernel","gaussian"),
                            upper_width=params.get("upper_width",0.5),
                            lower_width=params.get("lower_width",0.5),
                            trend_prop=params.get("trend_prop",0.0),
                            seasonal=params.get("seasonal",""),
                            drop_suspicious=params.get("drop_suspicious",True),
                            upper_detection=params.get("upper_detection",True),
                            lower_detection=params.get("lower_detection",True),
                            min_value=params.get("min_value",-1.0),
                            max_value=params.get("max_value",-1.0),
                            compensate_mode=params.get("compensate_mode","negative"),
                            compensate_coefficient=params.get("compensate_coefficient",0.1),
                            smooth_window=params.get("smooth_window",30),
                            fit_quantile=params.get("fit_quantile",0.5),
                            seasonal_anomaly_condition=params.get("seasonal_anomaly_condition","day,7,3"),
                            seasonal_anomaly_threshold=params.get("seasonal_anomaly_threshold",0.75),
                            upper_constant=params.get("upper_constant",-1.0),
                            lower_constant=params.get("lower_constant",-1.0))
        pre_value = np.zeros(len(train_data), dtype=float)
        upper = np.zeros(len(train_data), dtype=float)
        lower = np.zeros(len(train_data), dtype=float)
        anomaly = np.zeros(len(train_data), dtype=int)
        abnormality = np.zeros(len(train_data), dtype=int)
        for idx, data in train_data.iterrows():
            detect_result = spider_alg.fit(data)
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

    df = pd.read_csv('/Users/enze/ALG_Zsc/jax-algorithm/kernel/test/unseasonal_test_data.csv')
    time_field = "@timestamp"
    value_field = "@value"
    granularity = 300
    spiderAlg = Spider()
    params = {"time_field": time_field,
              "value_field": value_field,
              "granularity": granularity,
              "upper_width": 0.5,
              "lower_width": 0.5}
    result = Spider.run(df, params)
    import matplotlib.pyplot as plt

    result.index = pd.to_datetime(result[time_field])
    anomaly_index = result[result['anomaly'] == 1].index.tolist()
    plt.figure(figsize=(16, 8))
    plt.plot(result[value_field], color='k', label='value', alpha=0.75)
    plt.fill_between(result.index, result['upper'], result['lower'], color='#34A5DA', alpha=0.5, label='baseline')
    plt.scatter(anomaly_index, result.loc[anomaly_index, value_field], color='red', s=50, label='anomaly')
    plt.show()