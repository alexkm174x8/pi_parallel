# Setup Cluster MPI - NODES

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

## Parte 2.1 — Configurar Hostname

### Node1:
```bash
sudo hostnamectl set-hostname node1
```

### Node2:
```bash
sudo hostnamectl set-hostname node2
```

### Node3:
```bash
sudo hostnamectl set-hostname node3
```

Reiniciar después de cada cambio:
```bash
sudo reboot
```

---

## Parte 3 — Configurar IP Fija

1. Ver nombre de interfaz:
```bash
ip a
```

Busca algo como `enp0s3` o `eth0`

2. Abrir configuración netplan:
```bash
sudo nano /etc/netplan/01-network-manager-all.yaml
```

### Node1 - Configuración:
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

### Node2 - Configuración:
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

### Node3 - Configuración:
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

Probar conexión con otros nodos y la maestra:
```bash
ping master
ping node1
ping node2
ping node3
```

Para detener: `CTRL + C`

---

## Parte 6 — Desactivar Firewall

```bash
sudo ufw disable
```

---

## Verificar CPUs

```bash
nproc
```

> **Nota:** Anota este número para configurar correctamente el `machinefile` en la maestra.

---

## Verificar acceso a carpeta compartida y binario MPI

Asegúrate de que la carpeta del proyecto está montada en la misma ruta en todos los nodos (por ejemplo `/home/alejandro/pi_parallel/proyecto`).

Verifica que puedes listar el proyecto y que el ejecutable MPI está presente:

```bash
ls /home/alejandro/pi_parallel/proyecto
ls /home/alejandro/pi_parallel/proyecto/c_backend/bin/para_image_parra_mpi
```

Si el ejecutable no existe, pídele al maestro que compile y copie el binario a la carpeta compartida o ejecuta en el nodo:

```bash
mkdir -p /home/alejandro/pi_parallel/proyecto/c_backend/bin
mpicc -fopenmp /home/alejandro/pi_parallel/proyecto/c_backend/src/bmp_processor_mpi.c -o /home/alejandro/pi_parallel/proyecto/c_backend/bin/para_image_parra_mpi
```

Verifica también que las carpetas `img/` y `outputs/` son visibles:

```bash
ls /home/alejandro/pi_parallel/proyecto/img
ls /home/alejandro/pi_parallel/proyecto/outputs
```

## Apagar Nodo

```bash
sudo shutdown -h now
```
