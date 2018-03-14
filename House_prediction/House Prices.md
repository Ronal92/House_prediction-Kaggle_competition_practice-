House Prices: Advanced Regression Techniques.(V1)
============================================

	*이 글에 도움을 주신 분들!!*
	  > https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
	  > https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
	  

------------------------------------------------------------

# Using Library

- pandas
- seaborn
- numpy
- mapplot
- sklearn.preprocessing

# 순서(Data visualization -> 학습)

1. 주어진 데이터 이해하기
2. 데이터 간의 관계 파악
3. missing data, ourliar, get_dummi
3. Missing data 처리
4. 학습 & prediction

---------------------------------------------------------------

## 1. 주어진 데이터 이해하기

	직관적으로 집 가격을 결정할 수 있는 요인들 
		- OverallQual
		- MSSubClass(category) --> linear relationshep이 아니다
		- OverallCond 
		- LotArea(numerical)
		- YearBuilt --> 그래프상으로 linear relationship이 아니다.
		- GrLivArea
		- GarageArea
		- TotalBsmtSF


## 2. 데이터 간의 관계 파악

	<코드 : Relationship with numerical features>
		data = pd.concat( [df_train['SalePrice'], df_train['YearBuilt']] , axis = 1)
		data.plot.scatter(x='YearBuilt', y='SalePrice')

	<코드 : Relationship with categorical features>
		data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
		fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)

위 코드로 평면 그래프상에서 'SalePrice'와 다른 변수들과의 관계가 linear relationship인지 확인.

	<코드 : Correlation matrix>
		cols = ['SalePrice', 'OverallQual', 'MSSubClass', 'OverallCond', 'LotArea', 'YearBuilt', 'GrLivArea', 'GarageArea', 'TotalBsmtSF']
		corrmat = df_train[cols].corr()
		sns.heatmap(corrmat, vmax=.8, square=True)

이 코드는 변수들간의 상관관계를 heatmap으로 보고 이해할 수 있다.

	<현재까지의 코드 결과 분석>
	   - SalePrice랑 OverallCond, MSSubClass, YearBuilt는 관계없다.
	   - GrLivArea, GarageArea, TotalBsmtSF는 높은 correlation이 있다(정보가 중복될 수 있음..)

보다 정확한 상관관계를 위해 다음 코드를 이용하면 상관계수가 높은 상위 10개 변수들의 heatmap과 correlation 값을 알 수 있다.
	
	k = 10 #number of variables for heatmap
	cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
	cm = np.corrcoef(df_train[cols].values.T)
	sns.set(font_scale=1.25)
	hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

	--> OverallQual, GrLivArea, GarageArea, TotalBsmtSF, 1stFlrSF 변수가 주요 요인임!


## 3. Missing Data 처리

전체 observation 중에서 NA값이 들어있는 부분을 처리해주어야 한다.(여기서는 사용 안하는 건 과감히 버리기로!)

아래 코드는 전체 변수 중 null 혹은 nan이 들어있는 변수들과 그 비율을 알 수 있다.

	<코드>
	total = df_train.isnull().sum().sort_values(ascending=False)
	percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
	missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

	*분석*
		- 전체 데이터 중 15% 이상의 데이터가 missing value라면 복구 불가능하므로 사용하지 않는다.
		- 나머지 데이터 중에서 Garage와 관련된 변수들은 이미 GarageAreas에 정보가 반영되기 때문에 사용하지 않는다.
		- 'Electrical'은 missing value가 observation 하나이기 때문에 이 부분만 삭제한다.

분석을 바탕으로 전체 데이터에서 버려야 할 변수 혹은 observation을 버린다.

	<코드>
	df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
	df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
	df_train.isnull().sum().max()

## 4. Out liars
 
최종적으로 prediction을 위해서는 예측값의 정확도에 영향을 줄 수 있는 outliar들을 train data에서 지워 주어야 한다.
우선 표준정규분표를 사용해서 Out Liar로 지정할 기준을 설정한다.

	<코드 - 'SalePrice'의 표준정규분포 값>
	saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
	low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
	high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

SalePrice의 표준정규분표 결과, low_range에서는 문제 될게 없지만, high_range에서는 7 이상의 값들이 outliar로 볼 수 있다. OverallQual, GrLivArea, GarageAreas, TotalBsmtSF, 1stFlrSF를 독립변수로 SalePrice를 종속변수로 하는 scatter plot에서 7이상의 값들이 나올 경우 outliar로 볼 수 있다. 하지만 그래프 전체적으로 봤을 때, trend와 관련이 있는지를 조심해야 한다. 

	<코드>
	plt.scatter(df_train.GrLivArea, df_train.SalePrice, c = "blue", marker = "s")
	plt.title("Looking for outliers")
	plt.xlabel("GrLivArea")
	plt.ylabel("SalePrice")
	df_train = df_train[df_train.GrLivArea<4000]

## 5. Log transform

모델링을 할 때, target value의 분포가 정규 분포를 따르도록 하는 것이 좋다.
현재 target value 즉, SalePrice의 분포도를 보면, right skwed 된 것을 확인할 수 있다.

	<코드 - 'SalePrice'의 정규분표 표>
	sns.distplot(df_train['SalePrice'], fit=norm);
	fig = plt.figure()
	res = stats.probplot(df_train['SalePrice'], plot=plt)


	<코드 - log transformation>
	df_train.SalePrice = np.log1p(df_train.SalePrice)

## 6. feature engineering

train_data와 test_data를 합친 후 missing value와 Transforming, Skewed features 마지막으로 Getting Dummy로 데이터 처리를 마무리한다.	

## 7. Modeling
