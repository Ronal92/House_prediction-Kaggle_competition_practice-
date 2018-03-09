House Prices: Advanced Regression Techniques.(V1)
============================================

	*이 글에 도움을 주신 분들!!*
	  > https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python


------------------------------------------------------------

# Using Library

- pandas
- seaborn
- numpy
- mapplot

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

	<Relationship with numerical features>
		data = pd.concat( [df_train['SalePrice'], df_train['YearBuilt']] , axis = 1)
		data.plot.scatter(x='YearBuilt', y='SalePrice')

	<Relationship with categorical features>
		data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
		fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)

위 코드로 평면 그래프상에서 'SalePrice'와 다른 변수들과의 관계가 linear relationship인지 확인.

	<Correlation matrix>

		cols = ['SalePrice', 'OverallQual', 'MSSubClass', 'OverallCond', 'LotArea', 'YearBuilt', 'GrLivArea', 'GarageArea', 'TotalBsmtSF']
		corrmat = df_train[cols].corr()
		sns.heatmap(corrmat, vmax=.8, square=True)

이 코드는 변수들간의 상관관계를 heatmap으로 보고 이해할 수 있다.

현재까지의 코드 결과 분석
- SalePrice랑 OverallCond, MSSubClass, YearBuilt는 관계없다.
- GrLivArea, GarageArea, TotalBsmtSF는 높은 correlation이 있다(정보가 중복될 수 있음..)

다음으로 해야될  높은 상관계수를 가지는 상위 10개 요인들, outliar, missing data, get_dummy.....;;

