import pandas as pd
import sys

upward = '00462936.csv'
flows = 'wu03buk_oa_wz_v4.csv'


dfUp = pd.read_csv(upward)
dfUp = dfUp[['OutputArea', 'DataZone', 'InterZone']]
dfUp = dfUp.set_index('OutputArea')


dfMoves = pd.read_csv(flows, header=None)
dfMoves.columns = ['sourceOA', 'destOA', 'total', 'breakdown1', 'breakdown2', 'breakdown3']

# print(dfMoves)

withSourceDZ = dfMoves.merge(dfUp, how = 'inner', left_on='sourceOA', right_index=True )
withBothDZ = withSourceDZ.merge(dfUp, how = 'inner', left_on='destOA', right_index=True)

# print(withBothDZ)
withBothIZ = withBothDZ[['InterZone_x', 'InterZone_y', 'total']]
withBothIZ.columns = ['source_IZ', 'dest_IZ', 'weight']
withBothIZ = withBothIZ.groupby(['source_IZ', 'dest_IZ']).sum()

withBothDZ = withBothDZ[['DataZone_x', 'DataZone_y', 'total']]
withBothDZ.columns = ['source_DZ', 'dest_DZ', 'weight']
# print(withBothDZ)
# withBothDZ.to_csv('ungrouped_dz.csv')

withBothDZ = withBothDZ.groupby(['source_DZ', 'dest_DZ']).sum()

# print(withBothDZ)

withBothDZ.to_csv('wu03buk_oa_wz_v4_scottish_datazones.csv')
withBothIZ.to_csv('wu03buk_oa_wz_v4_scottish_interzones.csv')