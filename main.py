import clickhouse_connect
import sys
import json

def Click_house_cloud():
    CLICKHOUSE_CLOUD_HOSTNAME = 'nxefycxt62.eastus2.azure.clickhouse.cloud'
    CLICKHOUSE_CLOUD_USER = 'default'
    CLICKHOUSE_CLOUD_PASSWORD = 'ZoNOGOZPSv61~'

    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_CLOUD_HOSTNAME, 
        port=8443, 
        username=CLICKHOUSE_CLOUD_USER, 
        password=CLICKHOUSE_CLOUD_PASSWORD
    )
    return client

def Connector():
    client = Click_house_cloud()
    return "Connected"

if __name__ == "__main__":
    Connector()