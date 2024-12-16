class Personne():
    def __init__(self, id, revenuMensuel, montantDuPret, dureeDeEmploi, historiqueCredit):
        self.__id = id
        self.__revenuMensuel = revenuMensuel
        self.__montantDuPret = montantDuPret
        self.__dureeDeEmploi = dureeDeEmploi
        self.__historiqueCredit = historiqueCredit

    def getId(self):
        return self.__id

    def getRevenuMensuel(self):
        return self.__revenuMensuel
    
    def getMontantDuPret(self):
        return self.__montantDuPret
    
    def getDureeDeEmploi(self):
        return self.__dureeDeEmploi
    
    def getHistoriqueCredit(self):
        return self.__historiqueCredit
    
    def setId(self, id):
        self.__id = id

    def setRevenuMensuel(self, revenuMensuel):
        self.__revenuMensuel = revenuMensuel
    
    def setMontantDuPret(self, montantDuPret):
        self.__montantDuPret = montantDuPret
    
    def setDureeDeEmploi(self, dureeDeEmploi):
        self.__dureeDeEmploi = dureeDeEmploi
    
    def setHistoriqurCredit(self, historiqueCredit):
        self.__historiqueCredit = historiqueCredit