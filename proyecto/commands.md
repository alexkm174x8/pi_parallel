# Configuración de Cluster MPI

## Parte 1 — Actualizar Ubuntu

```bash
sudo apt update && sudo apt upgrade -y
```

## Parte 2 — Instalar OpenMPI y SSH

```bash
sudo apt install openmpi-bin openmpi-common libopenmpi-dev openssh-server -y
```

## Verificar IP Actual

```bash
ip a
```

---

## Parte 2.1 — Configurar Hostnames

### En la Maestra
```bash
sudo hostnamectl set-hostname master
```

### En Node1
```bash
sudo hostnamectl set-hostname node1
```

### En Node2
```bash
sudo hostnamectl set-hostname node2
```

### En Node3
```bash
sudo hostnamectl set-hostname node3
```

### Reiniciar TODAS las máquinas
```bash
sudo reboot
```

---

## Parte 3 — Configurar IPs Fijas

### En TODAS las computadoras

1. Ver nombre de interfaz:
```bash
ip a
```

Busca algo como `enp0s3` o `eth0`

2. Abrir configuración netplan:
```bash
sudo nano /etc/netplan/01-network-manager-all.yaml
```

### Configuración MAESTRA
```yaml
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    enp0s3:
      dhcp4: no
      addresses:
        - 192.168.1.10/24
      gateway4: 192.168.1.1
      nameservers:
        addresses:
          - 8.8.8.8
```

### Configuración NODE1
```yaml
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    enp0s3:
      dhcp4: no
      addresses:
        - 192.168.1.11/24
      gateway4: 192.168.1.1
      nameservers:
        addresses:
          - 8.8.8.8
```

### Configuración NODE2
```yaml
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    enp0s3:
      dhcp4: no
      addresses:
        - 192.168.1.12/24
      gateway4: 192.168.1.1
      nameservers:
        addresses:
          - 8.8.8.8
```

### Configuración NODE3
```yaml
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    enp0s3:
      dhcp4: no
      addresses:
        - 192.168.1.13/24
      gateway4: 192.168.1.1
      nameservers:
        addresses:
          - 8.8.8.8
```

3. Aplicar cambios:
```bash
sudo netplan apply
```

---

## Parte 4 — Configurar /etc/hosts

En TODAS las máquinas:
```bash
sudo nano /etc/hosts
```

Agregar las siguientes líneas:
```
192.168.1.10 master
192.168.1.11 node1
192.168.1.12 node2
192.168.1.13 node3
```

---

## Parte 5 — Probar Red

Desde la maestra:
```bash
ping node1
ping node2
ping node3
```

Para detener: `CTRL + C`

---

## Parte 6 — Configurar SSH Sin Password

### Solo en la MAESTRA

1. Generar llave SSH:
```bash
ssh-keygen -t rsa
```
Presiona ENTER a todo.

2. Copiar llave a nodos:
```bash
ssh-copy-id alejandro@node1
ssh-copy-id alejandro@node2
ssh-copy-id alejandro@node3
```

3. Probar acceso:
```bash
ssh node1
exit
ssh node2
exit
ssh node3
exit
```

---

## Parte 7 — Desactivar Firewall

En TODAS las máquinas:
```bash
sudo ufw disable
```

---

## Parte 8 — Crear Machinefile

### Solo en la MAESTRA

1. Crear archivo:
```bash
nano machinefile
```

2. Contenido:
```
master slots=2
node1 slots=2
node2 slots=2
node3 slots=2
```

Guardar con `CTRL + X` → `Y` → `ENTER`

---

## Parte 9 — Crear Programa MPI

### Solo en la MAESTRA

1. Crear archivo:
```bash
nano hello.c
```

2. Contenido:
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Get_processor_name(processor_name, &name_len);

    printf("Hola desde %s, proceso %d de %d\n",
           processor_name,
           world_rank,
           world_size);

    MPI_Finalize();
    return 0;
}
```

---

## Parte 10 — Compilar

```bash
mpicc hello.c -o hello
```

---

## Parte 11 — Ejecutar en el Cluster

```bash
mpirun --hostfile machinefile -np 8 ./hello
```

### Resultado Esperado:
```
Hola desde master, proceso 0 de 8
Hola desde node1, proceso 1 de 8
Hola desde node2, proceso 2 de 8
Hola desde node3, proceso 3 de 8
...
```

---

## Verificar CPUs

En cualquier máquina:
```bash
nproc
```

> **Nota:** Puedes aumentar `slots=` según núcleos reales

---

## Comando Útil para Ver Nodos Activos

Desde la maestra:
```bash
ssh node1 hostname
ssh node2 hostname
ssh node3 hostname
```

---

## Apagar Cluster

En cada máquina:
