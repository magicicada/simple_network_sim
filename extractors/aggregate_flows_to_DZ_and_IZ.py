import pandas as pd
import sys
import urllib.request

def main():
    url = 'https://www2.gov.scot/Resource/0046/00462936.csv'
    upward = '00462936.csv'
    urllib.request.urlretrieve(url, upward)
    # The file below is available from  http://wicid.ukdataservice.ac.uk/ to academics and local
    # government after making an account and agreeing to a EULA for access to safeguarded data
    flows = 'wu03buk_oa_wz_v4.csv'
    
    
    dfUp = pd.read_csv(upward)
    dfUp = dfUp[['OutputArea', 'DataZone', 'InterZone']]
    dfUp = dfUp.set_index('OutputArea')
    
    
    dfMoves = pd.read_csv(flows, names=['sourceOA', 'destOA', 'total', 'breakdown1', 'breakdown2', 'breakdown3'])
    
    withSourceDZ = dfMoves.merge(dfUp, how = 'inner', left_on='sourceOA', right_index=True )
    withBothDZ = withSourceDZ.merge(dfUp, how = 'inner', left_on='destOA', right_index=True)
    
    withBothIZ = withBothDZ[['InterZone_x', 'InterZone_y', 'total']]
    withBothIZ.columns = ['source_IZ', 'dest_IZ', 'weight']
    withBothIZ = withBothIZ.groupby(['source_IZ', 'dest_IZ']).sum()
    
    withBothDZ = withBothDZ[['DataZone_x', 'DataZone_y', 'total']]
    withBothDZ.columns = ['source_DZ', 'dest_DZ', 'weight']
    
    
    withBothDZ = withBothDZ.groupby(['source_DZ', 'dest_DZ']).sum()
    
    withBothDZ.to_csv('wu03buk_oa_wz_v4_scottish_datazones.csv')
    withBothIZ.to_csv('wu03buk_oa_wz_v4_scottish_interzones.csv')


if __name__ == '__main__':
    main()